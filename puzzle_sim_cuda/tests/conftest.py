# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for the CUDA test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# Make sure we can import both ``puzzle_sim`` (reference) and
# ``puzzle_sim_cuda`` (wrapper) regardless of where pytest is invoked from.
_REPO_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
_PKG_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
for p in (_REPO_ROOT, _PKG_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def _have_cuda() -> bool:
    return torch.cuda.is_available()


def _have_ext() -> bool:
    try:
        import puzzle_sim_cuda_ext  # noqa: F401
    except ImportError:
        return False
    return True


def _demo_dataset_dir(name: str) -> Path | None:
    """Return the path to a demo-data sample directory, or ``None`` if missing.

    Used by tests that need real images. We skip rather than fail when the
    data is unavailable so the suite still passes on machines that don't
    have the demo submodule checked out.
    """
    base = _REPO_ROOT / "PuzzleSim-demo-data" / "samples" / name
    if not base.is_dir():
        return None
    if not (base / "prior").is_dir() or not (base / "test").is_dir():
        return None
    return base


requires_cuda = pytest.mark.skipif(not _have_cuda(), reason="CUDA not available")
requires_ext = pytest.mark.skipif(not _have_ext(), reason="CUDA extension not built")


def requires_demo_data(name: str):
    """Skip marker for tests that need the named demo dataset."""
    return pytest.mark.skipif(
        _demo_dataset_dir(name) is None,
        reason=f"demo dataset {name!r} not available",
    )


@pytest.fixture(autouse=True)
def _seed_torch():
    """Make every test deterministic."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def garden_data() -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
    """Load the garden demo dataset onto the GPU once per session.

    Returns ``(priors, tests, test_names)`` or ``None`` if the data is missing.
    """
    base = _demo_dataset_dir("garden")
    if base is None:
        return None
    from PIL import Image
    import torchvision.transforms.functional as TF

    def _load_dir(dir_path: Path):
        names = sorted(p.name for p in dir_path.iterdir() if p.suffix.lower() == ".png")
        imgs = []
        for n in names:
            im = Image.open(dir_path / n).convert("RGB")
            imgs.append(TF.to_tensor(im).unsqueeze(0))
        return torch.cat(imgs, dim=0), names

    device = "cuda" if torch.cuda.is_available() else "cpu"
    priors, _ = _load_dir(base / "prior")
    tests, test_names = _load_dir(base / "test")
    return priors.to(device), tests.to(device), test_names
