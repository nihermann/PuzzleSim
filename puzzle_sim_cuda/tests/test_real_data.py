# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness tests on real image data.

These tests exercise ``PuzzleSimCUDA`` against ``puzzle_sim.PuzzleSim`` on the
garden dataset (36 priors + 4 test images, all 628x416 RGB).

They are slower than the random-tensor tests in :mod:`test_puzzlesim` so we
keep the matrix small (one or two test images, one or two configurations).
The point of these tests is to catch regressions that only surface on real
data - e.g. denormals, NaNs from edge pixels, or layout assumptions that
break on non-square inputs.
"""

from __future__ import annotations

import pytest
import torch

from .conftest import requires_cuda, requires_ext, requires_demo_data


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (
        (a - b).abs().max() / a.abs().max().clamp_min(1e-9)
    ).item()


@requires_cuda
@requires_ext
@requires_demo_data("garden")
@pytest.mark.parametrize("resize", [None, (256, 256), (128, 128)])
def test_garden_matches_reference(garden_data, resize):
    """Full PuzzleSim on the garden dataset must match the PyTorch reference.

    We use the first test image only (the others would just multiply runtime
    by 4 without exercising any new code path).
    """
    from puzzle_sim import PuzzleSim as RefPuzzleSim
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, tests, _ = garden_data
    img = tests[0:1]

    ref_model = RefPuzzleSim(priors, net_type="squeeze", resize=resize).to(priors.device).eval()
    cuda_model = PuzzleSimCUDA(priors, net_type="squeeze", resize=resize, precision="fp32")

    with torch.no_grad():
        ref_out = ref_model(img)
        cuda_out = cuda_model(img)

    assert ref_out.shape == cuda_out.shape == img.shape[-2:]
    assert torch.isfinite(cuda_out).all(), "CUDA output contains NaN/Inf"
    rel = _relative_error(ref_out, cuda_out)
    assert rel < 1e-4, f"resize={resize!r}: relative error too high: {rel:.3e}"


@requires_cuda
@requires_ext
@requires_demo_data("garden")
def test_garden_all_test_images(garden_data):
    """Every test image in the dataset should match the reference."""
    from puzzle_sim import PuzzleSim as RefPuzzleSim
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, tests, names = garden_data

    ref_model = RefPuzzleSim(priors, net_type="squeeze", resize=(256, 256)).to(priors.device).eval()
    cuda_model = PuzzleSimCUDA(priors, net_type="squeeze", resize=(256, 256), precision="fp32")

    for i, name in enumerate(names):
        img = tests[i:i + 1]
        with torch.no_grad():
            ref_out = ref_model(img)
            cuda_out = cuda_model(img)
        rel = _relative_error(ref_out, cuda_out)
        assert rel < 1e-4, f"{name}: relative error {rel:.3e}"


@requires_cuda
@requires_ext
@requires_demo_data("garden")
@pytest.mark.parametrize("net_type", ["squeeze", "alex", "vgg"])
def test_garden_all_backbones(garden_data, net_type):
    """All three backbones must work on the real-data path."""
    from puzzle_sim import PuzzleSim as RefPuzzleSim
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, tests, _ = garden_data
    img = tests[0:1]

    ref_model = RefPuzzleSim(priors, net_type=net_type, resize=(256, 256)).to(priors.device).eval()
    cuda_model = PuzzleSimCUDA(priors, net_type=net_type, resize=(256, 256), precision="fp32")

    with torch.no_grad():
        ref_out = ref_model(img)
        cuda_out = cuda_model(img)

    rel = _relative_error(ref_out, cuda_out)
    assert rel < 1e-4, f"net={net_type}: relative error {rel:.3e}"


@requires_cuda
@requires_ext
@requires_demo_data("garden")
def test_garden_output_is_a_valid_similarity_map(garden_data):
    """Sanity-check the metric values themselves.

    With default ``weights=(0.67, 0.2, 0.13)`` (sum == 1.0) and cosine
    similarities in [-1, 1], the per-pixel score is bounded in [-1, 1].
    On natural images we expect to be solidly in [0, 1].
    """
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, tests, _ = garden_data

    cuda_model = PuzzleSimCUDA(priors, net_type="squeeze", resize=(256, 256))
    with torch.no_grad():
        out = cuda_model(tests[0:1])

    assert torch.isfinite(out).all()
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0 + 1e-5
    assert out.mean().item() > 0.5, "expected high similarity on natural data"


@requires_cuda
@requires_ext
@requires_demo_data("garden")
def test_garden_self_image_high_similarity(garden_data):
    """An image that is itself one of the priors should produce a very high
    average similarity. This catches algorithmic mistakes that would otherwise
    only show up as small numerical disagreement with the reference."""
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, _, _ = garden_data

    cuda_model = PuzzleSimCUDA(priors, net_type="squeeze", resize=(256, 256))
    with torch.no_grad():
        out = cuda_model(priors[0:1])

    assert out.mean().item() > 0.95, (
        f"self-image average similarity should be ~1, got {out.mean().item():.3f}"
    )
