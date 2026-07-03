from __future__ import annotations

import os
from typing import Any

from setuptools import setup


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _detected_arch_list() -> list[str]:
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        return [item.strip() for item in env.split(";") if item.strip()]

    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return [f"{major}.{minor}"]
    return ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]


def _cuda_build_kwargs() -> dict[str, Any]:
    if not _truthy_env("PUZZLE_SIM_BUILD_CUDA"):
        return {}

    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except ImportError as exc:  # pragma: no cover - build-environment dependent
        raise RuntimeError(
            "PUZZLE_SIM_BUILD_CUDA=1 requires torch to be installed in the build environment. "
            "Use pip build isolation off for CUDA wheel builds."
        ) from exc

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(_detected_arch_list()))

    ext_modules = [
        CUDAExtension(
            name="puzzle_sim._cuda_ext",
            sources=[
                "src/puzzle_sim/cuda_ext/puzzle_sim_cuda.cpp",
                "src/puzzle_sim/cuda_ext/kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2"] if os.name == "nt" else ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-allow-unsupported-compiler",
                ],
            },
        )
    ]
    return {
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": BuildExtension},
        "zip_safe": False,
    }


setup(**_cuda_build_kwargs())
