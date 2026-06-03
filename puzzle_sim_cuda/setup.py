# SPDX-License-Identifier: Apache-2.0
"""Build script for the PuzzleSim CUDA extension.

Run ``pip install -e .`` from this directory, or ``python setup.py build_ext --inplace``
to drop the extension next to the Python sources.
"""

import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detected_arch() -> list[str]:
    """Return a TORCH_CUDA_ARCH_LIST-compatible list."""
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        return [a.strip() for a in env.split(";") if a.strip()]
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f"{major}.{minor}"]
    # Reasonable defaults for the GPUs we expect to see.
    return ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]


arch_list = _detected_arch()
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(arch_list))

ext_modules = [
    CUDAExtension(
        name="puzzle_sim_cuda_ext",
        sources=[
            "src/puzzle_sim_cuda.cpp",
            "src/fused_inference.cpp",
            "src/kernels.cu",
        ],
        extra_compile_args={
            "cxx": ["/O2"] if os.name == "nt" else ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                # Newer host compilers (e.g. MSVC 2026) are not yet on the
                # CUDA toolkit's supported list but still produce correct code.
                "-allow-unsupported-compiler",
            ],
        },
    ),
]

setup(
    name="puzzle_sim_cuda",
    version="0.2.0",
    description="CUDA-accelerated PuzzleSim implementation",
    packages=["puzzle_sim_cuda"],
    package_dir={"puzzle_sim_cuda": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
)
