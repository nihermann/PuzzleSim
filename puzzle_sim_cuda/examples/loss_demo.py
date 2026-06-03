#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Use PuzzleSim as a differentiable perceptual loss.

This optimises a *learnable image* so that its PuzzleSim similarity against a
reference distribution increases. The backbone weights and the reference are
frozen; gradients flow only to the image. It works with both the kernel and the
fused implementation (the fused model delegates the backward to an exact-FP32
kernel path under the hood).

Examples
--------
Synthetic reference, kernel mode::

    python loss_demo.py --steps 150

Fused mode against the garden demo set (if checked out)::

    python loss_demo.py --implementation fused --reference garden --steps 150

Optimise toward a folder of PNGs and save the result::

    python loss_demo.py --reference path/to/imgs --out optimized.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import torch

# [_REPO_ROOT, _PKG_DIR] so ``puzzle_sim`` + the ``puzzle_sim_cuda`` *package*
# resolve, and the co-located C++ extension is importable. See benchmark.py.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_PKG_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _synthetic_reference(n: int, size: int, device: str) -> torch.Tensor:
    """A few smooth low-frequency colour fields - cheap, structured targets."""
    torch.manual_seed(1234)
    ys = torch.linspace(0, 1, size, device=device)
    xs = torch.linspace(0, 1, size, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    refs = []
    for i in range(n):
        fx, fy = 1.0 + 2.0 * i, 1.5 + 1.5 * i
        r = 0.5 + 0.5 * torch.sin(2 * torch.pi * (fx * gx + 0.1 * i))
        g = 0.5 + 0.5 * torch.sin(2 * torch.pi * (fy * gy + 0.2 * i))
        b = 0.5 + 0.5 * torch.cos(2 * torch.pi * (fx * gx + fy * gy))
        refs.append(torch.stack([r, g, b], dim=0))
    return torch.stack(refs, dim=0).clamp(0, 1)


def _load_image_folder(path: str, size: int, device: str) -> Optional[torch.Tensor]:
    try:
        from PIL import Image
        import torchvision.transforms.functional as TF
    except ImportError:
        print("[warn] PIL/torchvision not available; cannot load images.")
        return None
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts)
    if not files:
        return None
    imgs = []
    for f in files:
        im = Image.open(os.path.join(path, f)).convert("RGB").resize((size, size))
        imgs.append(TF.to_tensor(im))
    return torch.stack(imgs, dim=0).to(device)


def _resolve_reference(name: str, n: int, size: int, device: str) -> torch.Tensor:
    if name == "synthetic":
        return _synthetic_reference(n, size, device)
    if name == "garden":
        garden = os.path.join(_REPO_ROOT, "PuzzleSim-demo-data", "samples", "garden", "prior")
        loaded = _load_image_folder(garden, size, device) if os.path.isdir(garden) else None
        if loaded is None:
            print("[warn] garden demo data not found; using synthetic reference.")
            return _synthetic_reference(n, size, device)
        return loaded
    loaded = _load_image_folder(name, size, device)
    if loaded is None:
        raise SystemExit(f"No usable images found at {name!r}")
    return loaded


def _build_model(implementation: str, reference: torch.Tensor, precision: str):
    from puzzle_sim_cuda import PuzzleSim

    return PuzzleSim(
        reference,
        net_type="squeeze",
        implementation=implementation,
        precision=precision,
    )


def _save_image(img: torch.Tensor, path: str) -> None:
    try:
        import torchvision.utils as vutils
    except ImportError:
        print(f"[warn] torchvision not available; cannot save {path}.")
        return
    vutils.save_image(img.clamp(0, 1), path)
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--implementation", choices=["kernel", "fused"], default="kernel")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16"], default="tf32",
                        help="Inference precision. The backward always uses exact FP32.")
    parser.add_argument("--reference", default="synthetic",
                        help="'synthetic', 'garden', or a folder of images.")
    parser.add_argument("--num-refs", type=int, default=4, help="Synthetic reference count.")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--out", default=None, help="Optional path to save the optimised image.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu (kernel mode only).")

    reference = _resolve_reference(args.reference, args.num_refs, args.size, args.device)
    print(f"reference: {tuple(reference.shape)} on {reference.device}")

    model = _build_model(args.implementation, reference, args.precision)

    img = torch.rand(1, 3, args.size, args.size, device=args.device, requires_grad=True)
    if args.out:
        _save_image(img.detach()[0], args.out.replace(".", "_init.", 1))

    opt = torch.optim.Adam([img], lr=args.lr)
    layers = tuple(args.layers)
    weights = tuple([1.0 / len(layers)] * len(layers))

    print(f"\nOptimising {args.implementation}/{args.precision} for {args.steps} steps "
          f"(layers={layers})\n" + "-" * 52)
    first = None
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        sim = model(img, layers=layers, weights=weights)
        loss = -sim.mean()  # maximise similarity == minimise negative similarity
        loss.backward()
        opt.step()
        with torch.no_grad():
            img.clamp_(0.0, 1.0)
        if step == 0:
            first = loss.item()
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:4d}   loss={loss.item():+.5f}   mean_sim={sim.mean().item():.5f}")

    last = loss.item()
    print("-" * 52)
    print(f"loss: {first:+.5f} -> {last:+.5f}   (reduced by {first - last:.5f})")
    assert last < first, "optimisation failed to reduce the loss"

    if args.out:
        _save_image(img.detach()[0], args.out)


if __name__ == "__main__":
    main()
