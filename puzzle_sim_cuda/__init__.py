# SPDX-License-Identifier: Apache-2.0
"""CUDA-accelerated PuzzleSim metric."""

from .puzzle_sim_cuda import CUDA_EXT_AVAILABLE, PuzzleSim, PuzzleSimCUDA
from .puzzle_sim_cuda_fused import PuzzleSimCUDAFused

__version__ = "0.2.0"
__all__ = [
    "PuzzleSim",
    "PuzzleSimCUDA",
    "PuzzleSimCUDAFused",
    "CUDA_EXT_AVAILABLE",
]


def get_version_info() -> dict:
    """Return a dict with package, torch and CUDA version info."""
    import torch

    info = {
        "package_version": __version__,
        "cuda_extension_available": CUDA_EXT_AVAILABLE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    return info
