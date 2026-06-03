# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness tests of ``PuzzleSimCUDA`` against the reference
PyTorch implementation in ``puzzle_sim.PuzzleSim``.

These tests pin ``precision="fp32"`` so any drift is purely from reduction-
order changes in our matmul + max chunking (not from TF32 / FP16 rounding).
The relaxed-precision paths are exercised in :mod:`test_tensor_cores`.
"""

from __future__ import annotations

import pytest
import torch

from .conftest import requires_cuda, requires_ext


def _make_inputs(num_refs: int, hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    refs = torch.rand(num_refs, 3, *hw, device="cuda")
    img = torch.rand(1, 3, *hw, device="cuda")
    return refs, img


def _run_ref(refs: torch.Tensor, img: torch.Tensor, **kwargs) -> torch.Tensor:
    from puzzle_sim import PuzzleSim as RefPuzzleSim
    model = RefPuzzleSim(refs, **kwargs).to(refs.device).eval()
    with torch.no_grad():
        return model(img)


def _run_cuda(refs: torch.Tensor, img: torch.Tensor, **kwargs) -> torch.Tensor:
    from puzzle_sim_cuda import PuzzleSimCUDA
    kwargs.setdefault("precision", "fp32")
    model = PuzzleSimCUDA(refs, **kwargs)
    with torch.no_grad():
        return model(img)


@requires_cuda
@requires_ext
@pytest.mark.parametrize("net_type", ["squeeze", "alex", "vgg"])
def test_matches_reference_on_random_inputs(net_type):
    """End-to-end agreement across the three supported backbones."""
    refs, img = _make_inputs(num_refs=5, hw=(96, 96))
    ref_out = _run_ref(refs, img, net_type=net_type)
    cuda_out = _run_cuda(refs, img, net_type=net_type)

    assert ref_out.shape == cuda_out.shape
    rel = (ref_out - cuda_out).abs().max() / ref_out.abs().max().clamp_min(1e-9)
    assert rel.item() < 1e-4, (
        f"net={net_type} relative error too high: {rel.item():.3e}"
    )


@requires_cuda
@requires_ext
@pytest.mark.parametrize("hw", [(64, 64), (96, 128), (128, 96), (160, 160)])
def test_matches_reference_across_input_shapes(hw):
    refs, img = _make_inputs(num_refs=4, hw=hw)
    ref_out = _run_ref(refs, img, net_type="squeeze")
    cuda_out = _run_cuda(refs, img, net_type="squeeze")
    assert ref_out.shape == (hw[0], hw[1])
    assert cuda_out.shape == (hw[0], hw[1])
    rel = (ref_out - cuda_out).abs().max() / ref_out.abs().max().clamp_min(1e-9)
    assert rel.item() < 1e-4


@requires_cuda
@requires_ext
def test_matches_reference_with_resize():
    refs, img = _make_inputs(num_refs=5, hw=(160, 160))
    ref_out = _run_ref(refs, img, net_type="squeeze", resize=(96, 96))
    cuda_out = _run_cuda(refs, img, net_type="squeeze", resize=(96, 96))
    assert ref_out.shape == cuda_out.shape
    rel = (ref_out - cuda_out).abs().max() / ref_out.abs().max().clamp_min(1e-9)
    assert rel.item() < 1e-4


@requires_cuda
@requires_ext
def test_self_image_is_high_similarity():
    """An image that is itself one of the references should score very high."""
    refs, _ = _make_inputs(num_refs=4, hw=(96, 96))
    img = refs[0:1].clone()
    out = _run_cuda(refs, img, net_type="squeeze")
    # The default weights sum to 1.0, so a "perfect" match would give ~1.0.
    assert out.mean().item() > 0.95


@requires_cuda
@requires_ext
def test_caches_reference_features_across_calls():
    """Second forward pass should reuse the packed reference cache."""
    from puzzle_sim_cuda import PuzzleSimCUDA

    refs, img = _make_inputs(num_refs=4, hw=(96, 96))
    model = PuzzleSimCUDA(refs, net_type="squeeze", precision="fp32")
    with torch.no_grad():
        out1 = model(img)
        # Mutate the underlying reference tensor. The cached features must
        # still be used (we intentionally do not invalidate the cache to
        # match the reference implementation's behaviour).
        refs.zero_()
        out2 = model(img)
    assert torch.allclose(out1, out2)


@requires_cuda
@requires_ext
def test_reduction_modes():
    """Mean reduction must equal sum reduction divided by the layer count."""
    refs, img = _make_inputs(num_refs=3, hw=(96, 96))
    from puzzle_sim_cuda import PuzzleSimCUDA
    model = PuzzleSimCUDA(refs, net_type="squeeze", precision="fp32")
    with torch.no_grad():
        s = model(img, reduction="sum")
        m = model(img, reduction="mean")
    # With default layers=(2,3,4), mean = sum / 3.
    assert torch.allclose(m, s / 3.0, atol=1e-6, rtol=1e-5)


@requires_cuda
@requires_ext
def test_weights_validation():
    refs, img = _make_inputs(num_refs=2, hw=(96, 96))
    from puzzle_sim_cuda import PuzzleSimCUDA
    model = PuzzleSimCUDA(refs, net_type="squeeze", precision="fp32")
    with pytest.raises(ValueError):
        model(img, layers=(2, 3, 4), weights=(0.5, 0.5))


@requires_cuda
@requires_ext
def test_invalid_net_type_raises():
    refs, _ = _make_inputs(num_refs=2, hw=(96, 96))
    from puzzle_sim_cuda import PuzzleSimCUDA
    with pytest.raises(ValueError):
        PuzzleSimCUDA(refs, net_type="not-a-real-net")
