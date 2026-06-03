// SPDX-License-Identifier: Apache-2.0
//
// PyTorch bindings for the PuzzleSim CUDA kernels.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>

// Forward declarations of the C-linkage kernel launchers from kernels.cu
extern "C" {
void launch_normalize_features(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps,
    cudaStream_t stream);

void launch_sim_max_tf32(
    const float* Q,
    const float* R,
    float* out,
    float* partial_max,
    int C_padded,
    int qHW_padded,
    int K_padded,
    int qHW,
    int K,
    cudaStream_t stream);

void launch_sim_max_fp16(
    const __half* Q,
    const __half* R,
    float* out,
    float* partial_max,
    int C_padded,
    int qHW_padded,
    int K_padded,
    int qHW,
    int K,
    cudaStream_t stream);

int sim_max_tile_m();
int sim_max_tile_n();

void launch_bilinear_upsample(
    const float* input,
    float* output,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    bool align_corners,
    cudaStream_t stream);
}

#define CHECK_CUDA(x)        TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x)       TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_HALF(x)        TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be float16")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_F32(x)   CHECK_CUDA(x); CHECK_FLOAT(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_F16(x)   CHECK_CUDA(x); CHECK_HALF(x); CHECK_CONTIGUOUS(x)

namespace {

inline cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream();
}

inline int round_up(int x, int mult) {
    return ((x + mult - 1) / mult) * mult;
}

inline std::pair<int, int> device_capability() {
    int dev = at::cuda::current_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    return {prop.major, prop.minor};
}

// Pads a (rows, cols) tensor to (new_rows, new_cols) with zeros.
// Returns the original tensor unchanged when no padding is needed.
torch::Tensor maybe_pad_2d(const torch::Tensor& t, int new_rows, int new_cols) {
    if (t.size(0) == new_rows && t.size(1) == new_cols) {
        return t;
    }
    auto padded = torch::zeros({new_rows, new_cols}, t.options());
    padded.narrow(0, 0, t.size(0)).narrow(1, 0, t.size(1)).copy_(t);
    return padded;
}

}  // namespace

// -----------------------------------------------------------------------------
// normalize_features
// -----------------------------------------------------------------------------
//
// Out-of-place L2 normalization across the channel axis.
// Input shape: (N, C, H, W). Output shape matches.
torch::Tensor normalize_features(torch::Tensor input, double eps) {
    CHECK_INPUT_F32(input);
    TORCH_CHECK(input.dim() == 4, "normalize_features expects 4D (N, C, H, W) input");

    auto output = torch::empty_like(input);

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    launch_normalize_features(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(eps),
        current_stream());

    return output;
}

// -----------------------------------------------------------------------------
// pack_reference - one-time layout transform
// -----------------------------------------------------------------------------
//
// Refs come in as (N, C, rH, rW). The similarity routine wants them as
// (C, N * rH * rW) so the contraction axis matches what torch.matmul expects.
// This is just a transpose+reshape but we expose it here so callers can do it
// once and cache the result.
torch::Tensor pack_reference(torch::Tensor refs) {
    CHECK_INPUT_F32(refs);
    TORCH_CHECK(refs.dim() == 4, "pack_reference expects 4D (N, C, H, W) input");

    const int N = refs.size(0);
    const int C = refs.size(1);
    const int H = refs.size(2);
    const int W = refs.size(3);

    return refs.transpose(0, 1).contiguous().reshape({C, N * H * W});
}

// -----------------------------------------------------------------------------
// sim_max_tf32 / sim_max_fp16 - tensor-core fused GEMM + per-row max
// -----------------------------------------------------------------------------

torch::Tensor sim_max_tf32(
    torch::Tensor query_packed,   // (C, qHW)
    torch::Tensor refs_packed,    // (C, K)
    std::vector<int64_t> query_hw)
{
    CHECK_INPUT_F32(query_packed);
    CHECK_INPUT_F32(refs_packed);
    TORCH_CHECK(query_packed.dim() == 2, "query_packed must be 2D (C, qHW)");
    TORCH_CHECK(refs_packed.dim() == 2,  "refs_packed must be 2D (C, K)");
    TORCH_CHECK(query_packed.size(0) == refs_packed.size(0),
                "Channel dim mismatch between query and refs");
    TORCH_CHECK(query_hw.size() == 2, "query_hw must be of length 2");

    auto cap = device_capability();
    TORCH_CHECK(cap.first >= 8,
                "sim_max_tf32 requires compute capability >= 8.0 (got ",
                cap.first, ".", cap.second, "); use fp16 or fp32 fallback");

    const int C = query_packed.size(0);
    const int qHW = query_packed.size(1);
    const int K = refs_packed.size(1);
    TORCH_CHECK(query_hw[0] * query_hw[1] == qHW,
                "query_hw does not match query_packed.shape[1]");

    constexpr int WMMA_K = 8;
    const int TC_TILE_M = sim_max_tile_m();
    const int TC_TILE_N = sim_max_tile_n();
    const int C_padded = round_up(C, WMMA_K);
    const int qHW_padded = round_up(qHW, TC_TILE_M);
    const int K_padded = round_up(K, TC_TILE_N);
    const int n_chunks = K_padded / TC_TILE_N;

    auto Q = maybe_pad_2d(query_packed, C_padded, qHW_padded);
    auto R = maybe_pad_2d(refs_packed,  C_padded, K_padded);

    auto out = torch::empty({query_hw[0], query_hw[1]}, query_packed.options());
    // Partial maxes from the split-K forward kernel; one float per
    // (q_padded, n_chunk).
    auto partial = torch::empty(
        {qHW_padded, n_chunks}, query_packed.options());

    launch_sim_max_tf32(
        Q.data_ptr<float>(),
        R.data_ptr<float>(),
        out.data_ptr<float>(),
        partial.data_ptr<float>(),
        C_padded, qHW_padded, K_padded, qHW, K,
        current_stream());

    return out;
}

torch::Tensor sim_max_fp16(
    torch::Tensor query_packed,   // (C, qHW) half
    torch::Tensor refs_packed,    // (C, K) half
    std::vector<int64_t> query_hw)
{
    CHECK_INPUT_F16(query_packed);
    CHECK_INPUT_F16(refs_packed);
    TORCH_CHECK(query_packed.dim() == 2, "query_packed must be 2D (C, qHW)");
    TORCH_CHECK(refs_packed.dim() == 2,  "refs_packed must be 2D (C, K)");
    TORCH_CHECK(query_packed.size(0) == refs_packed.size(0),
                "Channel dim mismatch between query and refs");
    TORCH_CHECK(query_hw.size() == 2, "query_hw must be of length 2");

    auto cap = device_capability();
    TORCH_CHECK(cap.first >= 7,
                "sim_max_fp16 requires compute capability >= 7.0 (got ",
                cap.first, ".", cap.second, ")");

    const int C = query_packed.size(0);
    const int qHW = query_packed.size(1);
    const int K = refs_packed.size(1);
    TORCH_CHECK(query_hw[0] * query_hw[1] == qHW,
                "query_hw does not match query_packed.shape[1]");

    constexpr int WMMA_K = 16;
    const int TC_TILE_M = sim_max_tile_m();
    const int TC_TILE_N = sim_max_tile_n();
    const int C_padded = round_up(C, WMMA_K);
    const int qHW_padded = round_up(qHW, TC_TILE_M);
    const int K_padded = round_up(K, TC_TILE_N);
    const int n_chunks = K_padded / TC_TILE_N;

    auto Q = maybe_pad_2d(query_packed, C_padded, qHW_padded);
    auto R = maybe_pad_2d(refs_packed,  C_padded, K_padded);

    auto float_opts = query_packed.options().dtype(at::kFloat);
    auto out = torch::empty({query_hw[0], query_hw[1]}, float_opts);
    auto partial = torch::empty({qHW_padded, n_chunks}, float_opts);

    launch_sim_max_fp16(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(R.data_ptr<at::Half>()),
        out.data_ptr<float>(),
        partial.data_ptr<float>(),
        C_padded, qHW_padded, K_padded, qHW, K,
        current_stream());

    return out;
}

// -----------------------------------------------------------------------------
// bilinear_upsample
// -----------------------------------------------------------------------------
//
// 2D bilinear upsample of a single (H, W) similarity map.
torch::Tensor bilinear_upsample(
    torch::Tensor input,
    std::vector<int64_t> output_size,
    bool align_corners)
{
    CHECK_INPUT_F32(input);
    TORCH_CHECK(input.dim() == 2, "bilinear_upsample expects 2D (H, W) input");
    TORCH_CHECK(output_size.size() == 2, "output_size must be of length 2");

    const int in_h = input.size(0);
    const int in_w = input.size(1);
    const int out_h = output_size[0];
    const int out_w = output_size[1];

    auto out = torch::empty({out_h, out_w}, input.options());

    launch_bilinear_upsample(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        in_h, in_w, out_h, out_w,
        align_corners,
        current_stream());

    return out;
}

// Registers the FusedSqueezeNet class. Implementation lives in
// fused_inference.cpp; forward-declared here so we don't have to drag that
// translation unit's includes into this header-only PYBIND11_MODULE block.
void register_fused_inference(pybind11::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated kernels for PuzzleSim";

    m.def("normalize_features", &normalize_features,
          "Channel-wise L2 normalization (CUDA)",
          py::arg("input"),
          py::arg("eps") = 1e-8);

    m.def("pack_reference", &pack_reference,
          "Transpose+flatten reference features into the (C, K) layout used "
          "for the similarity contraction",
          py::arg("refs"));

    m.def("sim_max_tf32", &sim_max_tf32,
          "Tensor-core fused GEMM + per-row max (TF32, sm_80+)",
          py::arg("query_packed"),
          py::arg("refs_packed"),
          py::arg("query_hw"));

    m.def("sim_max_fp16", &sim_max_fp16,
          "Tensor-core fused GEMM + per-row max (FP16 inputs, FP32 accumulator, sm_70+)",
          py::arg("query_packed"),
          py::arg("refs_packed"),
          py::arg("query_hw"));

    m.def("bilinear_upsample", &bilinear_upsample,
          "Bilinear upsample of a 2D similarity map (CUDA)",
          py::arg("input"),
          py::arg("output_size"),
          py::arg("align_corners") = true);

    register_fused_inference(m);
}
