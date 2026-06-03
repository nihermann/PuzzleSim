// SPDX-License-Identifier: Apache-2.0
//
// Fully fused PuzzleSim forward over the SqueezeNet1.1 backbone. Owns:
//   * pre-extracted, contiguous weight + bias tensors,
//   * pre-packed reference features (one per requested layer),
//   * the entire backbone -> normalize -> sim_max -> upsample -> weighted-sum
//     pipeline, executed back-to-back in a single C++ call so there is no
//     Python dispatch between the 8 Fire modules / 3 taps.
//
// Convolutions go through at::cudnn_convolution_relu, which is PyTorch's
// thin wrapper around cudnnConvolutionBiasActivationForward (single fused
// conv + bias + ReLU cuDNN op). Max-pool uses at::max_pool2d. The custom
// kernels (normalize / sim_max / bilinear_upsample) are reused from
// kernels.cu via the existing C-linkage launchers.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Forward declarations of the C-linkage kernel launchers from kernels.cu
extern "C" {
void launch_normalize_features(
    const float* input, float* output,
    int batch_size, int channels, int height, int width,
    float eps, cudaStream_t stream);

void launch_sim_max_tf32(
    const float* Q, const float* R,
    float* out, float* partial_max,
    int C_padded, int qHW_padded, int K_padded,
    int qHW, int K, cudaStream_t stream);

void launch_sim_max_fp16(
    const __half* Q, const __half* R,
    float* out, float* partial_max,
    int C_padded, int qHW_padded, int K_padded,
    int qHW, int K, cudaStream_t stream);

int sim_max_tile_m();
int sim_max_tile_n();

void launch_bilinear_upsample(
    const float* input, float* output,
    int in_h, int in_w, int out_h, int out_w,
    bool align_corners, cudaStream_t stream);
}

namespace fused {

namespace {

// SqueezeNet1.1 weight layout. The constructor receives a flat std::vector
// of tensors; these indices document where each weight lives.
//
//   [0]  conv0.weight  (64, 3, 3, 3)
//   [1]  conv0.bias    (64,)
//   [2 + 6*k + 0]  fire[k].squeeze.weight
//   [2 + 6*k + 1]  fire[k].squeeze.bias
//   [2 + 6*k + 2]  fire[k].expand1x1.weight
//   [2 + 6*k + 3]  fire[k].expand1x1.bias
//   [2 + 6*k + 4]  fire[k].expand3x3.weight
//   [2 + 6*k + 5]  fire[k].expand3x3.bias
// for k = 0..7.
constexpr int N_FIRES = 8;
constexpr int N_TENSORS = 2 + 6 * N_FIRES;  // 50

inline cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream();
}

inline int round_up(int x, int mult) {
    return ((x + mult - 1) / mult) * mult;
}

inline at::Tensor maybe_pad_2d(const at::Tensor& t, int new_rows, int new_cols) {
    if (t.size(0) == new_rows && t.size(1) == new_cols) {
        return t;
    }
    auto padded = torch::zeros({new_rows, new_cols}, t.options());
    padded.narrow(0, 0, t.size(0)).narrow(1, 0, t.size(1)).copy_(t);
    return padded;
}

inline at::Tensor conv_relu(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& b,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding)
{
    // cuDNN-backed conv + bias + ReLU in one op.
    return at::cudnn_convolution_relu(
        x, w, b,
        at::IntArrayRef(stride),
        at::IntArrayRef(padding),
        at::IntArrayRef({1, 1}),  // dilation
        /*groups=*/1);
}

// Fire module: squeeze (1x1) -> expand_1x1 (1x1) concat expand_3x3 (3x3).
inline at::Tensor fire_forward(
    const at::Tensor& x,
    const at::Tensor& sq_w, const at::Tensor& sq_b,
    const at::Tensor& e1_w, const at::Tensor& e1_b,
    const at::Tensor& e3_w, const at::Tensor& e3_b)
{
    auto sq = conv_relu(x, sq_w, sq_b, {1, 1}, {0, 0});
    auto e1 = conv_relu(sq, e1_w, e1_b, {1, 1}, {0, 0});
    auto e3 = conv_relu(sq, e3_w, e3_b, {1, 1}, {1, 1});
    return at::cat({e1, e3}, /*dim=*/1);
}

inline at::Tensor maxpool_3x3_s2_ceil(const at::Tensor& x) {
    return at::max_pool2d(
        x,
        /*kernel_size=*/at::IntArrayRef({3, 3}),
        /*stride=*/at::IntArrayRef({2, 2}),
        /*padding=*/at::IntArrayRef({0, 0}),
        /*dilation=*/at::IntArrayRef({1, 1}),
        /*ceil_mode=*/true);
}

// Channel-wise L2 normalize via the existing CUDA kernel.
inline at::Tensor normalize_features(const at::Tensor& x, double eps = 1e-8) {
    TORCH_CHECK(x.is_cuda(), "normalize_features requires a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "normalize_features expects 4D (N, C, H, W)");
    auto x_c = x.contiguous();
    auto out = torch::empty_like(x_c);
    launch_normalize_features(
        x_c.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(x_c.size(0)),
        static_cast<int>(x_c.size(1)),
        static_cast<int>(x_c.size(2)),
        static_cast<int>(x_c.size(3)),
        static_cast<float>(eps),
        current_stream());
    return out;
}

// Pack (N, C, H, W) -> (C, N*H*W). Same convention as the existing
// pack_reference kernel; kept as a thin wrapper here so we can call it
// from C++ on the L2-normalized backbone outputs.
inline at::Tensor pack_features(const at::Tensor& feats) {
    TORCH_CHECK(feats.dim() == 4, "pack_features expects 4D (N, C, H, W)");
    const auto N = feats.size(0);
    const auto C = feats.size(1);
    const auto H = feats.size(2);
    const auto W = feats.size(3);
    return feats.transpose(0, 1).contiguous().reshape({C, N * H * W});
}

// Tensor-core fused GEMM + per-row max. Mirrors the bindings in
// puzzle_sim_cuda.cpp; we duplicate here to keep this translation unit
// self-contained for the fused class.
inline at::Tensor sim_max_dispatch(
    const at::Tensor& query_packed,   // (C, qHW)
    const at::Tensor& refs_packed,    // (C, K)
    int qH, int qW,
    const std::string& precision)
{
    const int TC_TILE_M = sim_max_tile_m();
    const int TC_TILE_N = sim_max_tile_n();

    const int C = static_cast<int>(query_packed.size(0));
    const int qHW = static_cast<int>(query_packed.size(1));
    const int K = static_cast<int>(refs_packed.size(1));
    TORCH_CHECK(qH * qW == qHW, "query_hw does not match query_packed.shape[1]");

    if (precision == "fp32") {
        // Chunked matmul + max fallback. Mirrors the Python fallback exactly
        // but keeps everything on stream without round-tripping to Python.
        constexpr int64_t SCORE_CHUNK_BYTES = 256LL * 1024 * 1024;
        const int64_t bytes_per = query_packed.element_size();
        const int64_t max_chunk =
            std::max<int64_t>(1, SCORE_CHUNK_BYTES / std::max<int64_t>(1, qHW * bytes_per));
        const int64_t chunk = std::min<int64_t>(K, max_chunk);

        auto q_t = query_packed.transpose(0, 1);  // (qHW, C)
        if (chunk >= K) {
            auto scores = at::matmul(q_t, refs_packed);                 // (qHW, K)
            auto sim    = std::get<0>(scores.max(/*dim=*/1));
            return sim.reshape({qH, qW});
        }
        auto sim_map = torch::full({qHW},
                                   -std::numeric_limits<float>::infinity(),
                                   query_packed.options());
        for (int64_t k0 = 0; k0 < K; k0 += chunk) {
            int64_t k1 = std::min<int64_t>(k0 + chunk, K);
            auto partial = at::matmul(q_t, refs_packed.narrow(/*dim=*/1, k0, k1 - k0));
            auto partial_max = std::get<0>(partial.max(/*dim=*/1));
            sim_map = at::maximum(sim_map, partial_max);
        }
        return sim_map.reshape({qH, qW});
    }

    // Tensor-core path
    const bool fp16 = (precision == "fp16");
    const int WMMA_K = fp16 ? 16 : 8;
    const int C_padded   = round_up(C,   WMMA_K);
    const int qHW_padded = round_up(qHW, TC_TILE_M);
    const int K_padded   = round_up(K,   TC_TILE_N);
    const int n_chunks   = K_padded / TC_TILE_N;

    auto Q = maybe_pad_2d(query_packed, C_padded, qHW_padded);
    auto R = maybe_pad_2d(refs_packed,  C_padded, K_padded);

    auto float_opts = query_packed.options().dtype(at::kFloat);
    auto out = torch::empty({qH, qW}, float_opts);
    auto partial = torch::empty({qHW_padded, n_chunks}, float_opts);

    if (fp16) {
        launch_sim_max_fp16(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(R.data_ptr<at::Half>()),
            out.data_ptr<float>(),
            partial.data_ptr<float>(),
            C_padded, qHW_padded, K_padded, qHW, K,
            current_stream());
    } else {
        launch_sim_max_tf32(
            Q.data_ptr<float>(),
            R.data_ptr<float>(),
            out.data_ptr<float>(),
            partial.data_ptr<float>(),
            C_padded, qHW_padded, K_padded, qHW, K,
            current_stream());
    }
    return out;
}

inline at::Tensor bilinear_upsample(
    const at::Tensor& x, int64_t out_h, int64_t out_w, bool align_corners)
{
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kFloat,
                "bilinear_upsample expects a 2D float CUDA tensor");
    auto x_c = x.contiguous();
    auto out = torch::empty({out_h, out_w}, x_c.options());
    launch_bilinear_upsample(
        x_c.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(x_c.size(0)), static_cast<int>(x_c.size(1)),
        static_cast<int>(out_h), static_cast<int>(out_w),
        align_corners,
        current_stream());
    return out;
}

}  // namespace

// =====================================================================
// FusedSqueezeNet - holds packed weights + cached reference features,
// runs the full PuzzleSim pipeline in a single C++ call.
// =====================================================================
class FusedSqueezeNet {
public:
    FusedSqueezeNet(std::vector<at::Tensor> weights,
                    const std::string& precision)
        : w_(std::move(weights)), precision_(precision)
    {
        TORCH_CHECK(
            static_cast<int>(w_.size()) == N_TENSORS,
            "FusedSqueezeNet expects ", N_TENSORS, " weight tensors, got ", w_.size());
        for (size_t i = 0; i < w_.size(); ++i) {
            TORCH_CHECK(w_[i].is_cuda(), "weight ", i, " must be CUDA");
            TORCH_CHECK(w_[i].scalar_type() == at::kFloat,
                        "weight ", i, " must be float32; got ", w_[i].scalar_type());
            if (!w_[i].is_contiguous()) {
                w_[i] = w_[i].contiguous();
            }
        }
        validate_precision(precision_);
    }

    // Run the SqueezeNet backbone on `scaled_img` and return the
    // L2-normalized feature maps at the requested slice indices.
    //
    // `scaled_img` is expected to be (1, 3, H, W) and to have already
    // been through any scaling layer + optional resize (kept in Python).
    std::map<int64_t, at::Tensor> compute_features(
        at::Tensor scaled_img,
        std::vector<int64_t> layers)
    {
        TORCH_CHECK(scaled_img.is_cuda(), "scaled_img must be CUDA");
        if (scaled_img.dim() == 3) scaled_img = scaled_img.unsqueeze(0);
        TORCH_CHECK(scaled_img.dim() == 4 && scaled_img.size(1) == 3,
                    "scaled_img must be (N, 3, H, W)");

        std::map<int64_t, at::Tensor> tap_raw = forward_backbone(scaled_img, layers);

        std::map<int64_t, at::Tensor> out;
        for (auto& kv : tap_raw) {
            out[kv.first] = normalize_features(kv.second);
        }
        return out;
    }

    // Pack & cache reference features. Stores `(C, K)` packed tensors
    // (FP32 by default; FP16 mirror is built lazily for the fp16 precision
    // path the first time it is needed for a layer).
    //
    // `scaled_refs` is the reference distribution after scaling + resize,
    // shape `(N, 3, H, W)`.
    void set_reference(at::Tensor scaled_refs,
                       std::vector<int64_t> layers)
    {
        auto feats = compute_features(std::move(scaled_refs), layers);
        refs_packed_.clear();
        refs_packed_fp16_.clear();
        ref_hw_.clear();
        for (auto& kv : feats) {
            const auto& f = kv.second;
            refs_packed_[kv.first] = pack_features(f);
            ref_hw_[kv.first] = std::make_pair<int64_t, int64_t>(
                f.size(2), f.size(3));
        }
    }

    bool has_reference() const {
        return !refs_packed_.empty();
    }

    // The whole pipeline in one call. Returns a (out_h, out_w) similarity map.
    at::Tensor forward(at::Tensor scaled_img,
                       std::vector<int64_t> layers,
                       std::vector<double> weights,
                       const std::string& reduction,
                       std::vector<int64_t> out_hw)
    {
        TORCH_CHECK(!refs_packed_.empty(),
                    "set_reference must be called before forward");
        TORCH_CHECK(out_hw.size() == 2, "out_hw must be of length 2");
        TORCH_CHECK(weights.empty() || weights.size() == layers.size(),
                    "weights size must match layers size or be empty");

        auto feats = compute_features(std::move(scaled_img), layers);

        at::Tensor sum_map;
        bool first = true;
        for (size_t i = 0; i < layers.size(); ++i) {
            int64_t layer = layers[i];
            auto fit = feats.find(layer);
            auto rit = refs_packed_.find(layer);
            auto hit = ref_hw_.find(layer);
            TORCH_CHECK(fit != feats.end(), "missing query feats for layer ", layer);
            TORCH_CHECK(rit != refs_packed_.end(),
                        "missing packed refs for layer ", layer,
                        " - call set_reference with this layer first");
            (void)hit;  // ref_hw is kept for callers that want it; not needed here

            // Re-pack: (1, C, qH, qW) -> (C, qHW)
            const auto& q = fit->second;
            TORCH_CHECK(q.dim() == 4, "query feats must be 4D");
            const int qH = static_cast<int>(q.size(2));
            const int qW = static_cast<int>(q.size(3));
            const int C  = static_cast<int>(q.size(1));
            auto q_packed = q.reshape({C, qH * qW}).contiguous();

            // Picks the right refs tensor based on precision.
            at::Tensor r_packed = refs_for_layer(layer);
            if (precision_ == "fp16") {
                q_packed = q_packed.to(at::kHalf).contiguous();
            }

            auto sim_map = sim_max_dispatch(q_packed, r_packed, qH, qW, precision_);
            auto up_map  = bilinear_upsample(sim_map, out_hw[0], out_hw[1], /*align_corners=*/true);
            if (!weights.empty()) {
                up_map = up_map * weights[i];
            }
            if (first) {
                sum_map = up_map;
                first = false;
            } else {
                sum_map = sum_map + up_map;
            }
        }

        if (reduction == "mean" && !layers.empty()) {
            sum_map = sum_map / static_cast<double>(layers.size());
        }
        return sum_map;
    }

    std::string precision() const { return precision_; }

    void set_precision(const std::string& p) {
        validate_precision(p);
        precision_ = p;
        // Drop the FP16 cache so it gets rebuilt with the right dtype.
        refs_packed_fp16_.clear();
    }

private:
    static void validate_precision(const std::string& p) {
        TORCH_CHECK(p == "fp32" || p == "tf32" || p == "fp16",
                    "precision must be one of {fp32, tf32, fp16}; got ", p);
    }

    // Run the backbone and harvest the requested tap outputs.
    std::map<int64_t, at::Tensor> forward_backbone(
        at::Tensor x,
        const std::vector<int64_t>& layers)
    {
        // Layer index -> tap point in the topology:
        //   layer 0: conv0+relu
        //   layer 1: slice1 last fire (idx 1)
        //   layer 2: slice2 last fire (idx 3)
        //   layer 3: slice3 last fire (idx 4)
        //   layer 4: slice4 (idx 5)
        //   layer 5: slice5 (idx 6)
        //   layer 6: slice6 (idx 7)
        std::map<int64_t, bool> want;
        int64_t max_layer = -1;
        for (auto l : layers) {
            want[l] = true;
            if (l > max_layer) max_layer = l;
        }

        std::map<int64_t, at::Tensor> out;

        // Slice 0: conv0 (stride 2, padding 0) + ReLU.
        x = conv_relu(x, w_[0], w_[1], {2, 2}, {0, 0});
        if (want.count(0)) out[0] = x;
        if (max_layer == 0) return out;

        // Slice 1: maxpool, fire 0 (in=64), fire 1 (in=128, out=128).
        x = maxpool_3x3_s2_ceil(x);
        x = fire_at(x, 0);
        x = fire_at(x, 1);
        if (want.count(1)) out[1] = x;
        if (max_layer == 1) return out;

        // Slice 2: maxpool, fire 2 (in=128, out=256), fire 3 (in=256, out=256).
        x = maxpool_3x3_s2_ceil(x);
        x = fire_at(x, 2);
        x = fire_at(x, 3);
        if (want.count(2)) out[2] = x;
        if (max_layer == 2) return out;

        // Slice 3: maxpool, fire 4 (in=256, out=384).
        x = maxpool_3x3_s2_ceil(x);
        x = fire_at(x, 4);
        if (want.count(3)) out[3] = x;
        if (max_layer == 3) return out;

        // Slice 4: fire 5 (in=384, out=384).
        x = fire_at(x, 5);
        if (want.count(4)) out[4] = x;
        if (max_layer == 4) return out;

        // Slice 5: fire 6 (in=384, out=512).
        x = fire_at(x, 6);
        if (want.count(5)) out[5] = x;
        if (max_layer == 5) return out;

        // Slice 6: fire 7 (in=512, out=512).
        x = fire_at(x, 7);
        if (want.count(6)) out[6] = x;
        return out;
    }

    inline at::Tensor fire_at(const at::Tensor& x, int k) {
        const int base = 2 + 6 * k;
        return fire_forward(
            x,
            w_[base + 0], w_[base + 1],   // squeeze
            w_[base + 2], w_[base + 3],   // expand 1x1
            w_[base + 4], w_[base + 5]);  // expand 3x3
    }

    const at::Tensor& refs_for_layer(int64_t layer) {
        auto& fp32 = refs_packed_.at(layer);
        if (precision_ != "fp16") return fp32;
        auto it = refs_packed_fp16_.find(layer);
        if (it == refs_packed_fp16_.end()) {
            it = refs_packed_fp16_.emplace(
                layer, fp32.to(at::kHalf).contiguous()).first;
        }
        return it->second;
    }

    std::vector<at::Tensor> w_;
    std::map<int64_t, at::Tensor> refs_packed_;
    std::map<int64_t, at::Tensor> refs_packed_fp16_;
    std::map<int64_t, std::pair<int64_t, int64_t>> ref_hw_;
    std::string precision_;
};

}  // namespace fused

// -----------------------------------------------------------------------------
// pybind11 registration. Symbol-visible from the main puzzle_sim_cuda.cpp via
// a register_fused_inference(module&) helper invoked inside PYBIND11_MODULE.
// -----------------------------------------------------------------------------
void register_fused_inference(pybind11::module& m) {
    namespace py = pybind11;
    py::class_<fused::FusedSqueezeNet>(m, "FusedSqueezeNet")
        .def(py::init<std::vector<at::Tensor>, std::string>(),
             py::arg("weights"),
             py::arg("precision") = "tf32")
        .def("compute_features", &fused::FusedSqueezeNet::compute_features,
             py::arg("scaled_img"),
             py::arg("layers"),
             "Run scaling-free backbone + L2-normalize. Returns dict layer -> tensor.")
        .def("set_reference", &fused::FusedSqueezeNet::set_reference,
             py::arg("scaled_refs"),
             py::arg("layers"),
             "Compute and cache packed reference features.")
        .def("has_reference", &fused::FusedSqueezeNet::has_reference,
             "True once set_reference has been called.")
        .def("forward", &fused::FusedSqueezeNet::forward,
             py::arg("scaled_img"),
             py::arg("layers"),
             py::arg("weights"),
             py::arg("reduction") = "sum",
             py::arg("out_hw"),
             "Full PuzzleSim forward in one C++ call. Returns (H, W) sim map.")
        .def_property("precision",
                      &fused::FusedSqueezeNet::precision,
                      &fused::FusedSqueezeNet::set_precision);
}
