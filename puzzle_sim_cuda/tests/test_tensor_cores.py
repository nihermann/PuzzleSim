# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the tensor-core sim_max kernel.

These verify two things, mode by mode:

1. The kernel-level wrapper ``_sim_max(..., precision=p)`` agrees with the
   reference ``(Q.T @ R).max(dim=1)`` within the tolerance expected for the
   precision mode.
2. The full ``PuzzleSimCUDA`` model with ``precision={'tf32','fp16'}`` produces
   the same output as ``precision='fp32'`` within the same tolerance.

Tolerances:

* fp32:  atol=1e-5  (rounding floor)
* tf32:  atol=1e-3  (10-bit mantissa with FP32 accumulator)
* fp16:  atol=5e-3  (5-bit exponent + 10-bit mantissa with FP32 accumulator)
"""

from __future__ import annotations

import pytest
import torch

from .conftest import requires_cuda, requires_ext, requires_demo_data


_TOLERANCE = {"fp32": 1e-5, "tf32": 1e-3, "fp16": 5e-3}


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


def _reference_sim_max(Q: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return (Q.T @ R).max(dim=1).values


def _refs_for_precision(packed_fp32: torch.Tensor, precision: str) -> torch.Tensor:
    if precision == "fp16":
        return packed_fp32.half().contiguous()
    return packed_fp32


# ---------------------------------------------------------------------------
# Kernel-level: _sim_max in each mode
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("precision", ["fp32",
                                       pytest.param("tf32", marks=_SKIP_TF32),
                                       pytest.param("fp16", marks=_SKIP_FP16)])
@pytest.mark.parametrize("N,C,H,W", [
    (1,  32, 8,  8),       # small
    (10, 128, 16, 16),     # mid
    (1,  256, 32, 32),     # bigger qHW
    (3,  256, 13, 11),     # non-power-of-two
    (8,  512, 4,  4),
    (36, 256, 52, 78),     # garden-layer-2 scale
])
def test_sim_max_matches_matmul_for_each_precision(precision, N, C, H, W):
    from puzzle_sim_cuda.puzzle_sim_cuda import _sim_max, _pack_reference

    refs = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    q = torch.randn(C, H, W, device="cuda", dtype=torch.float32)
    refs = refs / torch.sqrt(1e-8 + refs.pow(2).sum(dim=1, keepdim=True))
    qn = q / torch.sqrt(1e-8 + q.pow(2).sum(dim=0, keepdim=True))

    packed = _pack_reference(refs.contiguous())
    refs_for_p = _refs_for_precision(packed, precision)

    out = _sim_max(qn, refs_for_p, (H, W), precision=precision)

    qp = qn.reshape(C, H * W).contiguous()
    refs_flat = refs.transpose(0, 1).contiguous().reshape(C, N * H * W)
    expected = _reference_sim_max(qp, refs_flat).reshape(H, W)

    atol = _TOLERANCE[precision]
    max_err = (out - expected).abs().max().item()
    assert max_err <= atol, (
        f"precision={precision}: max_err={max_err:.3e} exceeds tolerance {atol:.0e}"
    )


@requires_cuda
@requires_ext
@_SKIP_TF32
def test_sim_max_tf32_self_similarity_is_one():
    """When the query is a ref, the per-pixel max must be ~1 even for tf32."""
    from puzzle_sim_cuda.puzzle_sim_cuda import _sim_max, _pack_reference

    N, C, H, W = 3, 64, 8, 8
    refs = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    refs = refs / torch.sqrt(1e-8 + refs.pow(2).sum(dim=1, keepdim=True))
    packed = _pack_reference(refs.contiguous())

    out = _sim_max(refs[0], packed, (H, W), precision="tf32")
    assert torch.allclose(out, torch.ones_like(out), atol=_TOLERANCE["tf32"])


@requires_cuda
@requires_ext
@_SKIP_FP16
def test_sim_max_fp16_self_similarity_is_one():
    from puzzle_sim_cuda.puzzle_sim_cuda import _sim_max, _pack_reference

    N, C, H, W = 3, 64, 8, 8
    refs = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    refs = refs / torch.sqrt(1e-8 + refs.pow(2).sum(dim=1, keepdim=True))
    packed = _pack_reference(refs.contiguous()).half().contiguous()

    out = _sim_max(refs[0], packed, (H, W), precision="fp16")
    assert torch.allclose(out, torch.ones_like(out), atol=_TOLERANCE["fp16"])


# ---------------------------------------------------------------------------
# End-to-end: PuzzleSimCUDA in each mode against fp32 baseline
# ---------------------------------------------------------------------------

def _abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@requires_cuda
@requires_ext
@pytest.mark.parametrize("precision",
                         [pytest.param("tf32", marks=_SKIP_TF32),
                          pytest.param("fp16", marks=_SKIP_FP16)])
def test_puzzlesim_matches_fp32_baseline(precision):
    """End-to-end ``PuzzleSimCUDA`` agreement between fp32 and {tf32, fp16}."""
    from puzzle_sim_cuda import PuzzleSimCUDA

    refs = torch.rand(4, 3, 96, 96, device="cuda")
    img = torch.rand(1, 3, 96, 96, device="cuda")

    baseline = PuzzleSimCUDA(refs, net_type="squeeze", precision="fp32")
    model    = PuzzleSimCUDA(refs, net_type="squeeze", precision=precision)
    with torch.no_grad():
        out_fp32 = baseline(img)
        out_tc   = model(img)

    err = _abs_err(out_fp32, out_tc)
    # Relax the tolerance a little here vs the kernel-level test; the metric
    # sums three layers (each gets to drift by ~atol/3) and then bilinearly
    # upsamples, which doesn't add error but slightly smears it.
    atol = _TOLERANCE[precision] * 3
    assert err <= atol, f"precision={precision}: max abs err {err:.3e} > {atol:.0e}"


@requires_cuda
@requires_ext
@requires_demo_data("garden")
@pytest.mark.parametrize("precision",
                         [pytest.param("tf32", marks=_SKIP_TF32),
                          pytest.param("fp16", marks=_SKIP_FP16)])
def test_puzzlesim_garden_matches_fp32_baseline(garden_data, precision):
    """Same end-to-end agreement test on real images at the garden scale."""
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, tests, _ = garden_data
    img = tests[0:1]

    baseline = PuzzleSimCUDA(priors, net_type="squeeze", resize=(256, 256), precision="fp32")
    model    = PuzzleSimCUDA(priors, net_type="squeeze", resize=(256, 256), precision=precision)
    with torch.no_grad():
        out_fp32 = baseline(img)
        out_tc   = model(img)

    err = _abs_err(out_fp32, out_tc)
    atol = _TOLERANCE[precision] * 3
    assert torch.isfinite(out_tc).all()
    assert err <= atol, f"precision={precision}: max abs err {err:.3e} > {atol:.0e}"


# ---------------------------------------------------------------------------
# Behaviour guarantees: graceful fallback / explicit dispatch
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
def test_invalid_precision_raises():
    from puzzle_sim_cuda import PuzzleSimCUDA
    refs = torch.rand(2, 3, 64, 64, device="cuda")
    with pytest.raises(ValueError):
        PuzzleSimCUDA(refs, net_type="squeeze", precision="bf16")


@requires_cuda
@requires_ext
@_SKIP_FP16
def test_fp16_uses_fp16_cached_refs():
    """Calling forward with precision='fp16' must cache an FP16 copy of refs."""
    from puzzle_sim_cuda import PuzzleSimCUDA

    refs = torch.rand(3, 3, 64, 64, device="cuda")
    img = torch.rand(1, 3, 64, 64, device="cuda")
    model = PuzzleSimCUDA(refs, net_type="squeeze", precision="fp16")
    with torch.no_grad():
        _ = model(img)
    assert len(model._packed_refs_fp16) >= 1
    for v in model._packed_refs_fp16.values():
        assert v.dtype == torch.float16
