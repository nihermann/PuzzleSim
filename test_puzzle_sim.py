import os
import sys
from typing import get_args

import pytest
import torch

from puzzle_sim import PuzzleSim, find_best_matching_piece, NetType, VGGAlexSqueezeType, Dinov3Type

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.backends.mps.is_built() else device


def skip_dino_if_python_version_is_less_than_10(net_type: NetType) -> None:
    is_dino = net_type in get_args(Dinov3Type)
    if sys.version_info.minor < 10 and is_dino:
        pytest.skip("Dinov3 code requires Python>=3.10")


@pytest.mark.parametrize("mem_save", [True, False])
class TestFindBestMatchingPiece:
    def test_same_shape_in_out(self, mem_save: bool) -> None:
        priors = torch.rand(8, 3, 64, 64).to(device)
        test = priors[0]

        sims = find_best_matching_piece(priors, test, mem_save=mem_save)

        _, H, W = test.shape

        assert sims.shape == (H, W)

    def test_same_input_yields_max_sim_in_puzzle_matching(self, mem_save: bool) -> None:
        priors = torch.rand(8, 3, 64, 64).to(device)
        test = priors[0]

        sims = find_best_matching_piece(priors, test, mem_save=mem_save)

        assert torch.allclose(sims, torch.ones_like(sims))

    def test_spatial_propagation_puzzle_matching(self, mem_save: bool) -> None:
        priors = torch.zeros((8, 3, 64, 64)).to(device)
        test = torch.ones_like(priors[0])
        priors[0, :, 0, 0] = 1  # set one pixel to white

        sims = find_best_matching_piece(priors, test, mem_save=mem_save)

        assert torch.allclose(sims, torch.ones_like(sims))


@pytest.mark.parametrize("net_type", get_args(VGGAlexSqueezeType) + get_args(Dinov3Type))
class TestPuzzleSim:
    def test_same_input_yields_max_sim_in_puzzle_sim(self, net_type: NetType) -> None:
        skip_dino_if_python_version_is_less_than_10(net_type)

        priors = torch.rand(8, 3, 64, 64).to(device)
        test = priors[0]

        puzzle = PuzzleSim(priors, net_type=net_type, verbose=True)

        sims = puzzle(test, layers=(1, 2, 3))

        assert torch.allclose(sims, torch.ones_like(sims))

    def test_same_shape_in_out(self, net_type: NetType) -> None:
        skip_dino_if_python_version_is_less_than_10(net_type)

        priors = torch.rand(8, 3, 64, 64).to(device)
        test = priors[0]

        puzzle = PuzzleSim(priors, net_type=net_type, verbose=True)

        sims = puzzle(test, layers=(1, 2, 3))

        _, H, W = test.shape

        assert sims.shape == (H, W)

    def test_deterministic_output(self, net_type: NetType) -> None:
        skip_dino_if_python_version_is_less_than_10(net_type)

        priors = torch.rand(8, 3, 64, 64).to(device)
        test = priors[0]

        puzzle = PuzzleSim(priors, net_type=net_type, verbose=True)

        sims1 = puzzle(test, layers=(1, 2, 3))
        sims2 = puzzle(test, layers=(1, 2, 3))

        assert torch.allclose(sims1, sims2)


@pytest.mark.skipif(bool(os.environ.get("CI")) or not os.path.exists("src/puzzle_sim/dino_models/"), reason="Relies on local models so can't be run on CI")
@pytest.mark.parametrize("dino_type", get_args(Dinov3Type))
def test_correct_state_dict_loaded(dino_type: Dinov3Type):
    refs = torch.rand(8, 3, 64, 64).to(device)

    # using torch.hub causes rate limit issues in some environments, which can be fixed by this workaround
    # https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

    model = torch.hub.load(repo_or_dir="facebookresearch/dinov3", model=f"dinov3_{dino_type}",
        weights=f'src/puzzle_sim/dino_models/dinov3_{dino_type}_pretrain_lvd1689m{"-8aa4cbdd" if dino_type == "vitl16" else ""}.pth', ).to(
        device)
    model.eval()

    puzzle = PuzzleSim(refs, net_type=dino_type, verbose=True)

    tsd = model.state_dict()
    psd = puzzle.feature_extractor.model.state_dict()

    # fail fast
    assert tsd.keys() == psd.keys(), f"State dict keys do not match for net_type={dino_type}"

    actual_feats = puzzle.feature_extractor.compute_features(refs, n=model.n_blocks, normalize=True)
    puzzle.feature_extractor.model = model
    expected_feats = puzzle.feature_extractor.compute_features(refs, n=model.n_blocks, normalize=True)

    for key, val in model.state_dict().items():
        assert torch.allclose(tsd[key], psd[key]), \
            f"State dict values do not match for key={key} in net_type={dino_type}"

    for i in range(len(actual_feats)):
        assert torch.allclose(expected_feats[i], actual_feats[i]), \
            f"Features do not match for layer {i} in net_type={dino_type} with {torch.dist(expected_feats[i], actual_feats[i]):.3f} difference"
