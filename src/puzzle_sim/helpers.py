from typing import Tuple, Union
import torch
from torch import Tensor, nn


def upsample(in_tens: Tensor, out_hw: Tuple[int, ...] = (64, 64), align_corners: bool = False) -> Tensor:
    """Upsample input with bilinear interpolation."""
    while len(in_tens.shape) < 4:
        in_tens = in_tens.unsqueeze(0)
    return nn.Upsample(size=out_hw, mode="bilinear", align_corners=align_corners)(in_tens)


def resize_tensor(x: Tensor, size: Union[int, Tuple[int, int]] = 64, align_corners: bool = False) -> Tensor:
    """https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/sample_similarity_lpips.py#L127C22-L132."""
    if isinstance(size, int) and x.shape[-1] > size and x.shape[-2] > size:
        return torch.nn.functional.interpolate(x, (size, size), mode="area")
    return torch.nn.functional.interpolate(x, (size, size) if isinstance(size, int) else size, mode="bilinear", align_corners=align_corners, antialias=True)
