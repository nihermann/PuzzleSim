# SPDX-License-Identifier: Apache-2.0
"""Per-kernel correctness tests.

Each kernel is compared against an obvious PyTorch reference implementation
of the same operation. We use float32 throughout, so the tolerance is set just
above the rounding floor for the corresponding op.
"""

from __future__ import annotations

import pytest
import torch

from .conftest import requires_cuda, requires_ext


def _normalize_ref(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.sqrt(eps + x.pow(2).sum(dim=1, keepdim=True))


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("shape", [
    (1, 64, 16, 16),
    (4, 128, 32, 32),
    (8, 256, 8, 8),
    (1, 384, 64, 64),
    (3, 512, 7, 9),  # non-square spatial
])
def test_normalize_features_matches_pytorch(shape):
    import puzzle_sim_cuda_ext as ext

    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    ref = _normalize_ref(x)
    out = ext.normalize_features(x.contiguous(), 1e-8)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


@requires_cuda
@requires_ext
def test_normalize_features_unit_norm():
    """L2 norm along channels must be ~1 after normalization."""
    import puzzle_sim_cuda_ext as ext
    x = torch.randn(2, 64, 16, 16, device="cuda", dtype=torch.float32)
    out = ext.normalize_features(x.contiguous(), 1e-8)
    norms = out.pow(2).sum(dim=1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# pack_reference
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("shape", [(4, 32, 6, 6), (1, 128, 8, 8), (10, 256, 16, 16)])
def test_pack_reference_layout(shape):
    import puzzle_sim_cuda_ext as ext
    refs = torch.randn(*shape, device="cuda", dtype=torch.float32)
    packed = ext.pack_reference(refs.contiguous())
    N, C, H, W = shape
    assert packed.shape == (C, N * H * W)
    # Equivalent to transpose+flatten in torch.
    expected = refs.transpose(0, 1).contiguous().reshape(C, N * H * W)
    assert torch.equal(packed, expected)


# ---------------------------------------------------------------------------
# _sim_max (Python-side, cuBLAS-backed)
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("N,C,H,W", [
    (1,  32, 8,  8),
    (5,  64, 8,  8),
    (10, 128, 16, 16),
    (1,  256, 32, 32),
    (3,  256, 13, 11),   # awkward spatial sizes
    (8,  512, 4,  4),
    (36, 256, 52, 78),   # ~garden layer 2; large enough that chunking kicks in
])
def test_sim_max_matches_matmul(N, C, H, W):
    from puzzle_sim_cuda.puzzle_sim_cuda import _sim_max, _pack_reference

    refs = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    q = torch.randn(C, H, W, device="cuda", dtype=torch.float32)

    refs = refs / torch.sqrt(1e-8 + refs.pow(2).sum(dim=1, keepdim=True))
    qn = q / torch.sqrt(1e-8 + q.pow(2).sum(dim=0, keepdim=True))

    packed = _pack_reference(refs.contiguous())
    out = _sim_max(qn, packed, (H, W))

    # PyTorch reference: full GEMM + row-wise max.
    qp = qn.reshape(C, H * W).contiguous()
    refs_flat = refs.transpose(0, 1).contiguous().reshape(C, N * H * W)
    scores = qp.T @ refs_flat
    expected = scores.max(dim=1).values.reshape(H, W)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_sim_max_self_similarity_is_one():
    """When the query is one of the references, the per-pixel max must be 1
    (within float epsilon) because each query pixel matches itself exactly."""
    from puzzle_sim_cuda.puzzle_sim_cuda import _sim_max, _pack_reference

    N, C, H, W = 3, 64, 8, 8
    refs = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    refs = refs / torch.sqrt(1e-8 + refs.pow(2).sum(dim=1, keepdim=True))
    packed = _pack_reference(refs.contiguous())

    out = _sim_max(refs[0], packed, (H, W))

    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


# ---------------------------------------------------------------------------
# bilinear_upsample
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("in_hw,out_hw,align", [
    ((8, 8),   (64, 64),  True),
    ((16, 16), (128, 128), True),
    ((6, 9),   (32, 48),   True),
    ((8, 8),   (33, 41),   True),
    ((8, 8),   (64, 64),   False),
])
def test_bilinear_upsample_matches_torch(in_hw, out_hw, align):
    import puzzle_sim_cuda_ext as ext

    x = torch.randn(*in_hw, device="cuda", dtype=torch.float32)
    out = ext.bilinear_upsample(x.contiguous(), list(out_hw), align)
    expected = torch.nn.functional.interpolate(
        x.unsqueeze(0).unsqueeze(0),
        size=out_hw,
        mode="bilinear",
        align_corners=align,
    ).squeeze(0).squeeze(0)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
