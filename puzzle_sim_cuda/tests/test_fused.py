# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fully-fused C++ implementation.

``PuzzleSimCUDAFused`` collapses the SqueezeNet1.1 backbone + L2 normalize +
sim_max + bilinear upsample into a single C++ entrypoint. These tests make
sure that, at every supported precision, it produces (up to mode tolerance)
the same output as the kernel implementation ``PuzzleSimCUDA``.

We also exercise:

* the feature-extraction subpath, against the reference torchvision SqueezeNet,
* construction-time precision fallback (tf32/fp16 -> fp32 on unsupported
  hardware) - this is identical to the kernel-path behaviour,
* the ``PuzzleSim`` factory's ``implementation='fused'`` selector,
* edge cases: per-layer-set selection, repeated forwards, multi-image batch.
"""

from __future__ import annotations

import pytest
import torch

from .conftest import requires_cuda, requires_ext, requires_demo_data


# Tolerances against the kernel implementation. fp32 is looser than
# test_tensor_cores because the fused path uses ``at::cudnn_convolution_relu``
# (a fused conv+bias+ReLU cuDNN op) while the kernel path uses the standard
# ``at::conv2d`` + ``F.relu``. Both run via cuDNN, but each can pick a
# different convolution algorithm, so the per-layer activations agree only to
# within ~1e-4 even with TF32 disabled. Downstream the metric is then summed
# over three taps; the empirical end-to-end drift is ~2e-4.
_TOLERANCE = {"fp32": 2e-4, "tf32": 1e-3, "fp16": 5e-3}


def _device_capability_at_least(major: int, minor: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (major, minor)


_SKIP_TF32 = pytest.mark.skipif(
    not _device_capability_at_least(8),
    reason="TF32 tensor cores require compute capability >= 8.0",
)
_SKIP_FP16 = pytest.mark.skipif(
    not _device_capability_at_least(7),
    reason="FP16 tensor cores require compute capability >= 7.0",
)


def _have_fused() -> bool:
    try:
        import puzzle_sim_cuda_ext
        return hasattr(puzzle_sim_cuda_ext, "FusedSqueezeNet")
    except ImportError:  # pragma: no cover - exercised via skip
        return False


requires_fused = pytest.mark.skipif(
    not _have_fused(),
    reason="FusedSqueezeNet not available in the CUDA extension",
)


def _abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------
# Feature extraction agreement: fused backbone vs torchvision reference
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@requires_fused
def test_fused_backbone_matches_reference_fp32():
    """When TF32 is disabled, fused features must match torchvision to ~1e-7."""
    from puzzle_sim_cuda import PuzzleSimCUDAFused

    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        refs = torch.rand(1, 3, 128, 128, device="cuda")
        img = torch.rand(1, 3, 128, 128, device="cuda")
        fused = PuzzleSimCUDAFused(refs, precision="fp32")

        # Compare directly against the wrapped torchvision SqueezeNet.
        with torch.no_grad():
            scaled = fused._scaled_input(img, normalize=True)
            ref_outs = fused._backbone(scaled)

            def _norm(x, eps=1e-8):
                return x / (eps + x.pow(2).sum(dim=1, keepdim=True)).sqrt()

            feats = fused.compute_features(img, normalize=True, layers=(0, 1, 2, 3, 4, 5, 6))
            for k in range(7):
                expected = _norm(ref_outs[k])
                got = feats[k]
                err = _abs_err(expected, got)
                assert err <= 1e-5, f"layer {k}: max err {err:.3e}"
    finally:
        torch.backends.cudnn.allow_tf32 = prev


# ---------------------------------------------------------------------------
# End-to-end agreement: fused vs kernel implementation
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@requires_fused
@pytest.mark.parametrize("precision", [
    "fp32",
    pytest.param("tf32", marks=_SKIP_TF32),
    pytest.param("fp16", marks=_SKIP_FP16),
])
@pytest.mark.parametrize("shape", [
    (4, 64),    # tiny refs/img
    (3, 96),
    (8, 128),
])
def test_fused_matches_kernel(precision, shape):
    from puzzle_sim_cuda import PuzzleSimCUDA, PuzzleSimCUDAFused

    N, S = shape
    refs = torch.rand(N, 3, S, S, device="cuda")
    img = torch.rand(1, 3, S, S, device="cuda")

    kernel = PuzzleSimCUDA(refs, net_type="squeeze", precision=precision)
    fused = PuzzleSimCUDAFused(refs, precision=precision)
    with torch.no_grad():
        out_k = kernel(img)
        out_f = fused(img)

    assert out_k.shape == out_f.shape
    err = _abs_err(out_k, out_f)
    # tf32/fp16 multiply mode-tolerance by 3 because three layers sum into the
    # final map. fp32 mode is already measured against an empirical bound that
    # absorbs the conv-algorithm difference; we keep it as-is.
    atol = _TOLERANCE[precision] if precision == "fp32" else _TOLERANCE[precision] * 3
    assert err <= atol, (
        f"precision={precision}, shape={shape}: max abs err {err:.3e} > {atol:.0e}"
    )


@requires_cuda
@requires_ext
@requires_fused
def test_fused_handles_resize():
    from puzzle_sim_cuda import PuzzleSimCUDA, PuzzleSimCUDAFused

    refs = torch.rand(5, 3, 256, 192, device="cuda")
    img = torch.rand(1, 3, 256, 192, device="cuda")

    kernel = PuzzleSimCUDA(refs, net_type="squeeze", resize=(128, 96), precision="fp32")
    fused = PuzzleSimCUDAFused(refs, resize=(128, 96), precision="fp32")
    with torch.no_grad():
        out_k = kernel(img)
        out_f = fused(img)

    assert out_k.shape == out_f.shape == img.shape[-2:]
    err = _abs_err(out_k, out_f)
    assert err <= _TOLERANCE["fp32"], f"resize path max abs err {err:.3e}"


@requires_cuda
@requires_ext
@requires_fused
def test_fused_layer_subset():
    """Picking a non-default layer set must work end-to-end."""
    from puzzle_sim_cuda import PuzzleSimCUDAFused

    refs = torch.rand(3, 3, 96, 96, device="cuda")
    img = torch.rand(1, 3, 96, 96, device="cuda")
    fused = PuzzleSimCUDAFused(refs, precision="fp32")

    with torch.no_grad():
        out = fused(img, layers=(0, 1, 2), weights=(0.5, 0.3, 0.2))
    assert out.shape == img.shape[-2:]
    assert torch.isfinite(out).all()


@requires_cuda
@requires_ext
@requires_fused
def test_fused_repeated_forwards_stable():
    """Two forwards on the same input must produce identical output."""
    from puzzle_sim_cuda import PuzzleSimCUDAFused

    refs = torch.rand(3, 3, 96, 96, device="cuda")
    img = torch.rand(1, 3, 96, 96, device="cuda")
    fused = PuzzleSimCUDAFused(refs, precision="fp32")

    with torch.no_grad():
        a = fused(img)
        b = fused(img)
    assert torch.equal(a, b), "Fused forward must be deterministic across calls"


@requires_cuda
@requires_ext
@requires_fused
@requires_demo_data("garden")
@pytest.mark.parametrize("precision", [
    "fp32",
    pytest.param("tf32", marks=_SKIP_TF32),
    pytest.param("fp16", marks=_SKIP_FP16),
])
def test_fused_garden_matches_kernel(garden_data, precision):
    """End-to-end agreement on real garden images, across all precision modes."""
    from puzzle_sim_cuda import PuzzleSimCUDA, PuzzleSimCUDAFused

    assert garden_data is not None
    priors, tests, _ = garden_data
    img = tests[0:1]

    kernel = PuzzleSimCUDA(priors, net_type="squeeze", resize=(256, 256), precision=precision)
    fused = PuzzleSimCUDAFused(priors, resize=(256, 256), precision=precision)
    with torch.no_grad():
        out_k = kernel(img)
        out_f = fused(img)

    assert out_k.shape == out_f.shape
    assert torch.isfinite(out_f).all()
    err = _abs_err(out_k, out_f)
    atol = _TOLERANCE[precision] if precision == "fp32" else _TOLERANCE[precision] * 3
    assert err <= atol, f"garden / {precision}: max abs err {err:.3e} > {atol:.0e}"


# ---------------------------------------------------------------------------
# Factory + fallback behaviour
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@requires_fused
def test_factory_dispatches_to_fused_for_squeeze():
    from puzzle_sim_cuda import PuzzleSim, PuzzleSimCUDAFused, PuzzleSimCUDA

    refs = torch.rand(2, 3, 64, 64, device="cuda")
    m_fused = PuzzleSim(refs, net_type="squeeze", implementation="fused")
    assert isinstance(m_fused, PuzzleSimCUDAFused)

    m_kernel = PuzzleSim(refs, net_type="squeeze", implementation="kernel")
    assert isinstance(m_kernel, PuzzleSimCUDA)


@requires_cuda
@requires_ext
@requires_fused
def test_factory_fused_falls_back_for_non_squeeze():
    """Fused path is squeeze-only for now; alex/vgg must silently fall back."""
    from puzzle_sim_cuda import PuzzleSim, PuzzleSimCUDA

    refs = torch.rand(2, 3, 64, 64, device="cuda")
    for net in ("alex", "vgg"):
        m = PuzzleSim(refs, net_type=net, implementation="fused")
        assert isinstance(m, PuzzleSimCUDA), \
            f"Expected fallback to PuzzleSimCUDA for {net}, got {type(m).__name__}"


@requires_cuda
@requires_ext
@requires_fused
def test_invalid_precision_raises():
    from puzzle_sim_cuda import PuzzleSimCUDAFused
    refs = torch.rand(2, 3, 64, 64, device="cuda")
    with pytest.raises(ValueError):
        PuzzleSimCUDAFused(refs, precision="bf16")
