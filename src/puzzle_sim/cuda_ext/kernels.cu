// SPDX-License-Identifier: Apache-2.0
//
// CUDA kernels for PuzzleSim.
//
// Forward path:
//   * normalize_features_kernel - channel-wise L2 normalization with rsqrt.
//   * sim_max_tc_kernel<T,WMMA_K> - tensor-core GEMM + per-tile max reduction.
//       Split-K: grid is (qHW/TILE_M, K/TILE_N), each block emits one partial
//       max per query row in its N-chunk. A second kernel reduces along K.
//       T=float,WMMA_K=8  is the TF32 path (sm_80+).
//       T=__half,WMMA_K=16 is the FP16 path with FP32 accumulator (sm_70+).
//   * sim_max_reduce_kernel - reduces (qHW, n_chunks) partial maxes to (qHW,).
//   * bilinear_upsample_kernel - 2D bilinear upsample of a similarity map.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

namespace wmma = nvcuda::wmma;

// -----------------------------------------------------------------------------
// Feature normalization
// -----------------------------------------------------------------------------
//
// Computes  out[n, c, y, x] = in[n, c, y, x] / sqrt(eps + sum_c in[n, c, y, x]^2)
__global__ void normalize_features_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int spatial_size,
    float eps)
{
    int total_threads = batch_size * spatial_size;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_threads; idx += stride) {
        int batch = idx / spatial_size;
        int spatial = idx - batch * spatial_size;

        const float* in_ptr = input + batch * channels * spatial_size + spatial;
        float* out_ptr = output + batch * channels * spatial_size + spatial;

        float norm_sq = 0.0f;
        #pragma unroll 8
        for (int c = 0; c < channels; ++c) {
            float v = in_ptr[c * spatial_size];
            norm_sq += v * v;
        }

        float inv_norm = rsqrtf(norm_sq + eps);

        #pragma unroll 8
        for (int c = 0; c < channels; ++c) {
            out_ptr[c * spatial_size] = in_ptr[c * spatial_size] * inv_norm;
        }
    }
}

extern "C" void launch_normalize_features(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps,
    cudaStream_t stream)
{
    const int spatial = height * width;
    const int total = batch_size * spatial;
    const int block = 256;
    const int grid = min((total + block - 1) / block, 65535);
    normalize_features_kernel<<<grid, block, 0, stream>>>(
        input, output, batch_size, channels, spatial, eps);
}

// -----------------------------------------------------------------------------
// Tensor-core sim_max (split-K + reduce)
// -----------------------------------------------------------------------------
//
// Computes:  out[q] = max_k sum_c Q[c, q] * R[c, k]
//
// Layout requirements:
//   Q : (C_padded, qHW_padded), row-major in memory.
//   R : (C_padded, K_padded),   row-major in memory.
// "_padded" means rounded up to TILE_M, TILE_N, or WMMA_K as required; the
// padded region is filled with zeros by the host so dot-products are
// unaffected. The reduction additionally clips the max to ``K`` (so padded
// K columns cannot win the max).
//
// Strategy:
//   * Grid = (qHW_padded/TILE_M, K_padded/TILE_N). 4090 has 128 SMs; for the
//     garden-l2 scale that is ~70k blocks, plenty of parallelism.
//   * Each block computes ONE (TILE_M, TILE_N) score tile and reduces it to
//     (TILE_M,) partial max along its N-chunk.
//   * The host follows up with sim_max_reduce_kernel to collapse the per-block
//     partial maxes along the K dimension.
//   * Per block: 4 warps each own a 16-row band; per c_chunk both A and B are
//     staged into shared memory cooperatively so the four warps share the
//     loads from global. After accumulating all of C, fragments are stored to
//     a 64x68 (padding for bank-conflict-free reduce) shared-memory tile.

constexpr int TILE_M = 64;
constexpr int TILE_N = 128;
constexpr int WARPS_PER_BLOCK = 4;          // TILE_M / 16
constexpr int N_FRAGS_PER_WARP = 8;         // TILE_N / 16
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

// Per-precision fragment types and prepare() (rounding) hook.
template <typename T, int WMMA_K>
struct WmmaTraits;

template <>
struct WmmaTraits<float, 8> {
    using a_frag = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
    using b_frag = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
    using c_frag = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;
    using storage = float;

    template <typename F>
    static __device__ __forceinline__ void prepare(F& f) {
        #pragma unroll
        for (int i = 0; i < f.num_elements; i++) {
            f.x[i] = wmma::__float_to_tf32(f.x[i]);
        }
    }
};

template <>
struct WmmaTraits<__half, 16> {
    using a_frag = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
    using b_frag = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
    using c_frag = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using storage = __half;

    template <typename F>
    static __device__ __forceinline__ void prepare(F&) { /* fp16 needs no manual rounding */ }
};

template <typename T, int WMMA_K>
__global__ void sim_max_tc_kernel(
    const T* __restrict__ Q,            // (C_padded, qHW_padded)
    const T* __restrict__ R,            // (C_padded, K_padded)
    float* __restrict__ partial_max,    // (qHW_padded, n_chunks)
    int C_padded,
    int qHW_padded,
    int K_padded,
    int K,
    int n_chunks)
{
    using Traits = WmmaTraits<T, WMMA_K>;
    using a_frag_t = typename Traits::a_frag;
    using b_frag_t = typename Traits::b_frag;
    using c_frag_t = typename Traits::c_frag;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;             // 0..3
    const int block_m = blockIdx.x * TILE_M;
    const int n_chunk_idx = blockIdx.y;
    const int n_start = n_chunk_idx * TILE_N;
    if (block_m >= qHW_padded) return;
    const int warp_m_offset = warp_id * 16;

    // Shared memory staging for A (Q tile) and B (R tile).
    __shared__ T A_shared[TILE_M * WMMA_K];
    __shared__ T B_shared[WMMA_K * TILE_N];

    // +4 padding eliminates the 32-way bank conflict the row-wise reduce would
    // otherwise hit (with LD == TILE_N, every row would start in bank 0). Must
    // also be a multiple of 4 (wmma::store_matrix_sync constraint).
    // Each warp owns its own 16 x LD slot in shmem to halve the total
    // footprint vs a single TILE_M x LD region - the per-warp slots aren't
    // shared across warps, so we can fit more concurrent blocks per SM.
    constexpr int LD = TILE_N + 4;
    __shared__ float score_tiles[WARPS_PER_BLOCK * 16 * LD];

    c_frag_t acc[N_FRAGS_PER_WARP];
    #pragma unroll
    for (int i = 0; i < N_FRAGS_PER_WARP; i++) {
        wmma::fill_fragment(acc[i], 0.0f);
    }

    for (int c_chunk = 0; c_chunk < C_padded; c_chunk += WMMA_K) {
        // Cooperative load of A_shared (TILE_M x WMMA_K, row-major) from
        // Q (C_padded x qHW_padded, row-major). Walks one C-slice per
        // iteration so consecutive threads load consecutive q values for
        // the same channel - coalesced 128-byte transactions per warp.
        #pragma unroll
        for (int i = tid; i < TILE_M * WMMA_K; i += THREADS_PER_BLOCK) {
            int c_local = i / TILE_M;
            int q_local = i - c_local * TILE_M;
            A_shared[q_local * WMMA_K + c_local] =
                Q[(c_chunk + c_local) * qHW_padded + block_m + q_local];
        }

        // Cooperative load of B_shared (WMMA_K x TILE_N, row-major).
        #pragma unroll
        for (int i = tid; i < WMMA_K * TILE_N; i += THREADS_PER_BLOCK) {
            int c_local = i / TILE_N;
            int n_local = i - c_local * TILE_N;
            B_shared[i] = R[(c_chunk + c_local) * K_padded + n_start + n_local];
        }
        __syncthreads();

        a_frag_t a_frag;
        wmma::load_matrix_sync(
            a_frag,
            &A_shared[warp_m_offset * WMMA_K],
            WMMA_K);
        Traits::prepare(a_frag);

        #pragma unroll
        for (int n_frag = 0; n_frag < N_FRAGS_PER_WARP; n_frag++) {
            b_frag_t b_frag;
            wmma::load_matrix_sync(
                b_frag,
                &B_shared[n_frag * 16],
                TILE_N);
            Traits::prepare(b_frag);

            wmma::mma_sync(acc[n_frag], a_frag, b_frag, acc[n_frag]);
        }

        __syncthreads();  // before A_shared/B_shared are overwritten next chunk
    }

    // Store this warp's 16xTILE_N band into its per-warp shmem slot.
    float* warp_tile = &score_tiles[warp_id * 16 * LD];
    #pragma unroll
    for (int n_frag = 0; n_frag < N_FRAGS_PER_WARP; n_frag++) {
        wmma::store_matrix_sync(
            &warp_tile[n_frag * 16],
            acc[n_frag],
            LD,
            wmma::mem_row_major);
    }
    // No __syncthreads needed: the reduce below only reads this warp's slot,
    // and the per-warp slots do not overlap.

    // Row-wise max over the TILE_N columns of this N-chunk, masked to actual K.
    // Each warp reduces its own 16 rows; lane 0..15 of each warp handle a row.
    const int lane = tid & 31;
    if (lane < 16) {
        int cols_remaining = K - n_start;
        int n_limit = (cols_remaining >= TILE_N) ? TILE_N
                    : (cols_remaining > 0       ? cols_remaining : 0);
        const float* row = &warp_tile[lane * LD];
        float row_max = -CUDART_INF_F;
        #pragma unroll 8
        for (int n = 0; n < n_limit; n++) {
            row_max = fmaxf(row_max, row[n]);
        }
        const int global_q = block_m + warp_m_offset + lane;
        partial_max[global_q * n_chunks + n_chunk_idx] = row_max;
    }
}

// Reduce (qHW_padded, n_chunks) partial maxes to (qHW,) final maxes.
// One block per query row, threads cooperate over the n_chunks axis.
__global__ void sim_max_reduce_kernel(
    const float* __restrict__ partial_max,    // (qHW_padded, n_chunks)
    float* __restrict__ out,                  // (qHW,)
    int qHW,
    int n_chunks)
{
    const int q = blockIdx.x;
    if (q >= qHW) return;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    extern __shared__ float buf[];
    float local_max = -CUDART_INF_F;
    const float* row = partial_max + (size_t)q * n_chunks;
    for (int i = tid; i < n_chunks; i += threads) {
        local_max = fmaxf(local_max, row[i]);
    }
    buf[tid] = local_max;
    __syncthreads();

    // Tree-style reduction within the block.
    for (int s = threads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            buf[tid] = fmaxf(buf[tid], buf[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) out[q] = buf[0];
}

// Host helper that launches the forward + reduce pair.
template <typename T, int WMMA_K>
static void launch_sim_max_tc_impl(
    const T* Q, const T* R, float* out,
    float* partial_max,
    int C_padded, int qHW_padded, int K_padded, int qHW, int K,
    cudaStream_t stream)
{
    const int n_chunks = K_padded / TILE_N;
    dim3 grid(qHW_padded / TILE_M, n_chunks);
    sim_max_tc_kernel<T, WMMA_K><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        Q, R, partial_max, C_padded, qHW_padded, K_padded, K, n_chunks);

    // Reduce. 128 threads per row; shared mem = 128 * 4 = 512 B.
    const int reduce_threads = 128;
    const int reduce_shmem = reduce_threads * sizeof(float);
    sim_max_reduce_kernel<<<qHW, reduce_threads, reduce_shmem, stream>>>(
        partial_max, out, qHW, n_chunks);
}

extern "C" void launch_sim_max_tf32(
    const float* Q, const float* R, float* out,
    float* partial_max,
    int C_padded, int qHW_padded, int K_padded, int qHW, int K,
    cudaStream_t stream)
{
    launch_sim_max_tc_impl<float, 8>(
        Q, R, out, partial_max,
        C_padded, qHW_padded, K_padded, qHW, K, stream);
}

extern "C" void launch_sim_max_fp16(
    const __half* Q, const __half* R, float* out,
    float* partial_max,
    int C_padded, int qHW_padded, int K_padded, int qHW, int K,
    cudaStream_t stream)
{
    launch_sim_max_tc_impl<__half, 16>(
        Q, R, out, partial_max,
        C_padded, qHW_padded, K_padded, qHW, K, stream);
}

// Exposes the kernel-side tile sizes to the host bindings so the padding math
// has a single source of truth (see TC_TILE_M/TC_TILE_N use in the .cpp).
extern "C" int sim_max_tile_m() { return TILE_M; }
extern "C" int sim_max_tile_n() { return TILE_N; }

// -----------------------------------------------------------------------------
// Bilinear upsample
// -----------------------------------------------------------------------------
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    bool align_corners)
{
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= out_h || out_x >= out_w) return;

    float scale_y, scale_x;
    if (align_corners) {
        scale_y = (out_h > 1) ? (float)(in_h - 1) / (out_h - 1) : 0.0f;
        scale_x = (out_w > 1) ? (float)(in_w - 1) / (out_w - 1) : 0.0f;
    } else {
        scale_y = (float)in_h / (float)out_h;
        scale_x = (float)in_w / (float)out_w;
    }

    float src_y = align_corners ? out_y * scale_y : (out_y + 0.5f) * scale_y - 0.5f;
    float src_x = align_corners ? out_x * scale_x : (out_x + 0.5f) * scale_x - 0.5f;

    src_y = fmaxf(0.0f, fminf(src_y, (float)(in_h - 1)));
    src_x = fmaxf(0.0f, fminf(src_x, (float)(in_w - 1)));

    int y0 = (int)floorf(src_y);
    int x0 = (int)floorf(src_x);
    int y1 = min(y0 + 1, in_h - 1);
    int x1 = min(x0 + 1, in_w - 1);

    float dy = src_y - y0;
    float dx = src_x - x0;

    float v00 = input[y0 * in_w + x0];
    float v01 = input[y0 * in_w + x1];
    float v10 = input[y1 * in_w + x0];
    float v11 = input[y1 * in_w + x1];

    float v0 = v00 * (1.0f - dx) + v01 * dx;
    float v1 = v10 * (1.0f - dx) + v11 * dx;
    output[out_y * out_w + out_x] = v0 * (1.0f - dy) + v1 * dy;
}

extern "C" void launch_bilinear_upsample(
    const float* input,
    float* output,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    bool align_corners,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
    bilinear_upsample_kernel<<<grid, block, 0, stream>>>(
        input, output, in_h, in_w, out_h, out_w, align_corners);
}
