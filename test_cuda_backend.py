import pytest
import torch

from puzzle_sim import PuzzleSim, find_best_matching_piece


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_forced_cuda_backend_requires_cuda_tensors() -> None:
    refs = torch.rand(2, 3, 8, 8)
    img = refs[0]

    with pytest.raises(RuntimeError, match="CUDA backend requires CUDA float32 tensors"):
        find_best_matching_piece(refs, img, backend="cuda")


@requires_cuda
def test_find_best_matching_piece_auto_matches_torch_backend() -> None:
    refs = torch.rand(3, 5, 8, 7, device="cuda")
    img = torch.rand(5, 8, 7, device="cuda")

    expected = find_best_matching_piece(refs, img, backend="torch")
    actual = find_best_matching_piece(refs, img, backend="auto")

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_puzzlesim_auto_backend_matches_torch_backend() -> None:
    refs = torch.rand(2, 3, 64, 64, device="cuda")
    img = torch.rand(1, 3, 64, 64, device="cuda")

    torch_model = PuzzleSim(refs, net_type="squeeze", backend="torch")
    auto_model = PuzzleSim(refs, net_type="squeeze", backend="auto")

    with torch.no_grad():
        expected = torch_model(img, layers=(1, 2, 3))
        actual = auto_model(img, layers=(1, 2, 3))

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
