from typing import Literal

import torch
import pytest
from puzzle_sim import PuzzleSim, find_best_matching_piece


device = 'gpu' if torch.cuda.is_available() else 'cpu'


@pytest.mark.parametrize("mem_save", [True, False])
def test_same_input_yields_max_sim_in_puzzle_matching(mem_save: bool) -> None:
    priors = torch.rand(8, 3, 64, 64).to(device)
    test = priors[0]

    sims = find_best_matching_piece(priors, test, mem_save=mem_save)

    assert torch.allclose(sims, torch.ones_like(sims))


@pytest.mark.parametrize("mem_save", [True, False])
def test_spatial_propagation_puzzle_matching(mem_save: bool) -> None:
    priors = torch.zeros((8, 3, 64, 64)).to(device)
    test = torch.ones_like(priors[0])
    priors[0,:,0,0] = 1  # set one pixel to white

    sims = find_best_matching_piece(priors, test, mem_save=mem_save)

    assert torch.allclose(sims, torch.ones_like(sims))



@pytest.mark.parametrize("net_type", ["vgg", "alex", "squeeze"])
def test_same_input_yields_max_sim_in_puzzle_sim(net_type: Literal['vgg', 'alex', 'squeeze']) -> None:
    priors = torch.rand(8, 3, 64, 64).to(device)
    test = priors[0]

    puzzle = PuzzleSim(priors, net_type=net_type)

    sims = puzzle(test)

    assert torch.allclose(sims, torch.ones_like(sims))