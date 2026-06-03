#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark suite for PuzzleSim CUDA against the PyTorch reference.

The script reports:

* ``backbone_only``: the cost of the torchvision backbone + scaling. This is
  shared by both implementations and acts as an effective floor on total time.
* ``ref total`` / ``cuda total``: end-to-end forward pass on the reference and
  CUDA implementations, including the backbone.
* ``ref core``  / ``cuda core``: end-to-end minus the backbone cost - this is
  the portion that the CUDA kernels are actually replacing.
* ``total speedup``: ref/cuda for end-to-end time.
* ``core speedup``: ref-core/cuda-core - the meaningful number for the kernel
  work itself.

Run with ``python benchmark.py``. Use ``--scenarios all`` for a larger sweep,
``--csv path.csv`` to dump results, ``--reps N`` to change timing samples.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Tuple

import torch

# Allow running this script from anywhere. We need:
#   * ``_REPO_ROOT`` on sys.path so ``import puzzle_sim`` (reference impl)
#     resolves AND ``import puzzle_sim_cuda`` resolves to the *package*
#     (not the inner module of the same name);
#   * ``_PKG_DIR`` on sys.path so the ``puzzle_sim_cuda_ext`` C++ extension
#     - which lives inside the package directory - is importable.
# Order matters: _REPO_ROOT must come first so the package wins. We insert
# _PKG_DIR first then _REPO_ROOT so the final layout is [_REPO_ROOT, _PKG_DIR, ...].
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_PKG_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(fn: Callable[[], object], reps: int, warmup: int) -> Tuple[float, float]:
    """Return (best seconds, peak GPU MB) over ``reps`` runs.

    Uses ``torch.cuda.Event`` for device-side timing. We report the best
    (minimum) sample because Windows WDDM and background GPU work routinely
    add 100ms-class delays on top of the actual kernel work; the median is
    not a stable signal of what the kernels themselves cost. The minimum
    approximates the no-contention runtime - which is the meaningful number
    when comparing two implementations.
    """
    for _ in range(warmup):
        fn()
    _cuda_sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    samples: List[float] = []
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(reps):
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 1000.0)
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)

    best = min(samples)
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available() else 0.0
    )
    return best, peak_mb


@dataclass
class ScenarioResult:
    scenario: str
    precision: str
    implementation: str
    backbone_ms: float
    ref_total_ms: float
    cuda_total_ms: float
    ref_core_ms: float
    cuda_core_ms: float
    total_speedup: float
    core_speedup: float
    ref_peak_mb: float
    cuda_peak_mb: float

    def as_row(self) -> List[str]:
        # When cuda_core is well below the measurement noise floor (i.e. the
        # CUDA work is essentially free compared to the backbone) the ratio is
        # not meaningful - report '>=N' as a lower bound instead.
        if self.cuda_core_ms < 0.05:
            core_cell = f"   >> "
        else:
            core_cell = f"{self.core_speedup:5.2f}x"
        return [
            self.scenario,
            self.implementation,
            self.precision,
            f"{self.backbone_ms:7.2f}",
            f"{self.ref_total_ms:7.2f}",
            f"{self.cuda_total_ms:7.2f}",
            f"{self.ref_core_ms:7.2f}",
            f"{self.cuda_core_ms:7.2f}",
            f"{self.total_speedup:5.2f}x",
            core_cell,
            f"{self.ref_peak_mb:6.0f}",
            f"{self.cuda_peak_mb:6.0f}",
        ]


def _measure_backbone(
    model,
    img: torch.Tensor,
    resize: Optional[Tuple[int, int]],
    reps: int,
    warmup: int,
) -> float:
    """Time the scaling + (optional) resize + backbone forward.

    Both implementations share this code path, so this measurement is
    representative of the per-call backbone cost in the full forward.
    The caller is expected to pass a model whose backbone has already been
    exercised at least once (cuDNN tunes its kernels on first input shape).
    """
    from puzzle_sim import _resize_tensor as torch_resize

    def go():
        with torch.no_grad():
            x = 2 * img - 1
            x = model.scaling_layer(x)
            if resize is not None:
                x = torch_resize(x, size=resize, align_corners=True)
            model.net.forward(x)
    t, _ = _bench(go, reps=reps, warmup=warmup)
    return t * 1000


def _load_dataset(name: str, device: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load a sample dataset from ``PuzzleSim-demo-data/samples/<name>``.

    Returns ``(priors, tests)`` or ``None`` if the data is unavailable.
    """
    from pathlib import Path
    base = Path(_REPO_ROOT) / "PuzzleSim-demo-data" / "samples" / name
    if not (base / "prior").is_dir() or not (base / "test").is_dir():
        return None
    from PIL import Image
    import torchvision.transforms.functional as TF

    def _load_dir(p: "Path"):
        names = sorted(x for x in p.iterdir() if x.suffix.lower() == ".png")
        imgs = [TF.to_tensor(Image.open(x).convert("RGB")).unsqueeze(0) for x in names]
        return torch.cat(imgs, dim=0)

    priors = _load_dir(base / "prior").to(device)
    tests = _load_dir(base / "test").to(device)
    return priors, tests


def run_scenario(
    name: str,
    num_refs: int,
    ref_hw: Tuple[int, int],
    test_hw: Tuple[int, int],
    net_type: str = "squeeze",
    resize: Optional[Tuple[int, int]] = None,
    reps: int = 20,
    warmup: int = 3,
    precisions: Tuple[str, ...] = ("tf32",),
    implementations: Tuple[str, ...] = ("kernel",),
) -> List[ScenarioResult]:
    refs = torch.rand(num_refs, 3, *ref_hw, device="cuda")
    img = torch.rand(1, 3, *test_hw, device="cuda")

    return _run_pair(
        name, refs, img, net_type=net_type, resize=resize,
        reps=reps, warmup=warmup, precisions=precisions,
        implementations=implementations,
    )


def _make_cuda_model(impl: str, refs: torch.Tensor, net_type: str,
                     resize: Optional[Tuple[int, int]], precision: str):
    """Construct a CUDA model for the given implementation choice.

    ``impl='fused'`` only supports the SqueezeNet backbone today; for any
    other ``net_type`` the call returns ``None`` so the caller can skip the
    fused row.
    """
    from puzzle_sim_cuda import PuzzleSimCUDA, PuzzleSimCUDAFused

    if impl == "kernel":
        return PuzzleSimCUDA(refs, net_type=net_type, resize=resize, precision=precision)
    if impl == "fused":
        if net_type != "squeeze":
            return None
        return PuzzleSimCUDAFused(refs, resize=resize, precision=precision)
    raise ValueError(f"Unknown implementation: {impl!r}")


def _run_pair(
    name: str,
    refs: torch.Tensor,
    img: torch.Tensor,
    net_type: str,
    resize: Optional[Tuple[int, int]],
    reps: int,
    warmup: int,
    precisions: Tuple[str, ...],
    implementations: Tuple[str, ...] = ("kernel",),
) -> List[ScenarioResult]:
    """Benchmark a reference/CUDA model pair across implementations + precisions.

    Reference is timed once (it does not have a precision/implementation
    setting). For each (implementation, precision) pair, the CUDA model is
    built fresh, warmed, then timed. The backbone measurement is shared
    across all CUDA configurations.
    """
    from puzzle_sim import PuzzleSim as RefPuzzleSim
    from puzzle_sim_cuda import PuzzleSimCUDA

    ref_model = RefPuzzleSim(refs, net_type=net_type, resize=resize).to(refs.device).eval()

    # Warm and time the backbone once (same architecture across precisions).
    warm_cuda = PuzzleSimCUDA(refs, net_type=net_type, resize=resize, precision="fp32")
    with torch.no_grad():
        for _ in range(warmup):
            ref_model(img)
            warm_cuda(img)
    _cuda_sync()

    backbone_ms = _measure_backbone(
        warm_cuda, img, resize=resize, reps=reps, warmup=2,
    )

    def run_ref():
        with torch.no_grad():
            ref_model(img)
    ref_t, ref_mem = _bench(run_ref, reps=reps, warmup=2)
    ref_ms = ref_t * 1000

    out: List[ScenarioResult] = []
    for impl in implementations:
        for precision in precisions:
            cuda_model = _make_cuda_model(impl, refs, net_type, resize, precision)
            if cuda_model is None:
                # fused requested but unsupported for this net_type - skip silently.
                continue
            with torch.no_grad():
                for _ in range(warmup):
                    cuda_model(img)
            _cuda_sync()

            def run_cuda():
                with torch.no_grad():
                    cuda_model(img)
            cuda_t, cuda_mem = _bench(run_cuda, reps=reps, warmup=2)
            cuda_ms = cuda_t * 1000
            ref_core = max(0.0, ref_ms - backbone_ms)
            cuda_core = max(0.0, cuda_ms - backbone_ms)
            out.append(ScenarioResult(
                scenario=name,
                precision=precision,
                implementation=impl,
                backbone_ms=backbone_ms,
                ref_total_ms=ref_ms,
                cuda_total_ms=cuda_ms,
                ref_core_ms=ref_core,
                cuda_core_ms=cuda_core,
                total_speedup=ref_ms / max(cuda_ms, 1e-6),
                core_speedup=ref_core / max(cuda_core, 1e-6),
                ref_peak_mb=ref_mem,
                cuda_peak_mb=cuda_mem,
            ))
    return out


QUICK_SCENARIOS = [
    dict(name="5x64",   num_refs=5,  ref_hw=(64, 64),   test_hw=(64, 64)),
    dict(name="10x128", num_refs=10, ref_hw=(128, 128), test_hw=(128, 128)),
    dict(name="20x256(resize=128)", num_refs=20, ref_hw=(256, 256), test_hw=(256, 256), resize=(128, 128)),
]


FULL_SCENARIOS = QUICK_SCENARIOS + [
    dict(name="50x128", num_refs=50, ref_hw=(128, 128), test_hw=(128, 128)),
    dict(name="50x256(resize=128)", num_refs=50, ref_hw=(256, 256), test_hw=(256, 256), resize=(128, 128)),
    dict(name="10x256-alex", num_refs=10, ref_hw=(256, 256), test_hw=(256, 256), resize=(128, 128), net_type="alex"),
    dict(name="10x256-vgg", num_refs=10, ref_hw=(256, 256), test_hw=(256, 256), resize=(128, 128), net_type="vgg"),
]


# Real-data scenarios run on the named demo dataset. They are skipped at
# runtime if the data is not present.
REAL_SCENARIOS = [
    dict(name="garden(36x628x416)",            dataset="garden", resize=None,         net_type="squeeze"),
    dict(name="garden(36x628x416,resize=256)", dataset="garden", resize=(256, 256),   net_type="squeeze"),
    dict(name="garden(36x628x416,resize=128)", dataset="garden", resize=(128, 128),   net_type="squeeze"),
    dict(name="garden-alex(resize=256)",       dataset="garden", resize=(256, 256),   net_type="alex"),
    dict(name="garden-vgg(resize=128)",        dataset="garden", resize=(128, 128),   net_type="vgg"),
]


_COL_WIDTHS = [34, 7, 6, 11, 9, 10, 8, 9, 8, 7, 7, 7]


def _print_header() -> None:
    headers = [
        "scenario", "impl", "prec", "backbone_ms", "ref_total", "cuda_total",
        "ref_core", "cuda_core", "total_sp", "core_sp",
        "ref_MB", "cuda_MB",
    ]
    print("  ".join(h.ljust(w) for h, w in zip(headers, _COL_WIDTHS)))
    print("-" * (sum(_COL_WIDTHS) + len(_COL_WIDTHS) * 2))


def _print_result(r: ScenarioResult) -> None:
    cells = r.as_row()
    print("  ".join(c.ljust(w) for c, w in zip(cells, _COL_WIDTHS)))


def _run_real_scenario(
    name: str,
    dataset: str,
    resize: Optional[Tuple[int, int]],
    net_type: str,
    reps: int,
    warmup: int,
    precisions: Tuple[str, ...],
    implementations: Tuple[str, ...] = ("kernel",),
) -> Optional[List[ScenarioResult]]:
    """Run a benchmark scenario on a real demo dataset.

    Returns ``None`` if the dataset is unavailable so the caller can skip.
    """
    data = _load_dataset(dataset, "cuda")
    if data is None:
        return None
    refs, tests = data
    img = tests[0:1]
    return _run_pair(
        name, refs, img, net_type=net_type, resize=resize,
        reps=reps, warmup=warmup, precisions=precisions,
        implementations=implementations,
    )


def _parse_precisions(spec: str) -> Tuple[str, ...]:
    """Parse a comma-separated precision spec.

    ``"all"`` expands to ``("fp32", "tf32", "fp16")``. Individual items must be
    one of the supported precisions.
    """
    if spec == "all":
        return ("fp32", "tf32", "fp16")
    valid = {"fp32", "tf32", "fp16"}
    out: List[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok not in valid:
            raise SystemExit(
                f"--precision: {tok!r} is not one of {sorted(valid)} (or 'all')"
            )
        out.append(tok)
    return tuple(out)


def _parse_implementations(spec: str) -> Tuple[str, ...]:
    if spec == "all":
        return ("kernel", "fused")
    valid = {"kernel", "fused"}
    out: List[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok not in valid:
            raise SystemExit(
                f"--implementation: {tok!r} is not one of {sorted(valid)} (or 'all')"
            )
        out.append(tok)
    return tuple(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        choices=["quick", "all", "real", "all+real"],
        default="quick",
        help=(
            "Which scenario set to run. 'quick' / 'all' use synthetic random "
            "tensors; 'real' uses the demo datasets (garden); 'all+real' is "
            "the union."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="tf32",
        help=(
            "Comma-separated precisions to benchmark, or 'all' for fp32,tf32,fp16. "
            "Each scenario is emitted as one row per precision."
        ),
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="kernel",
        help=(
            "Comma-separated implementations to benchmark, or 'all' for "
            "kernel,fused. 'fused' is only valid for net_type='squeeze' and "
            "is silently skipped for other backbones."
        ),
    )
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to write the results table to.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available.")
        return 1

    precisions = _parse_precisions(args.precision)
    implementations = _parse_implementations(args.implementation)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"precisions: {', '.join(precisions)}")
    print(f"implementations: {', '.join(implementations)}")
    print()

    synthetic = []
    real = []
    if args.scenarios in ("quick",):
        synthetic = QUICK_SCENARIOS
    elif args.scenarios in ("all",):
        synthetic = FULL_SCENARIOS
    elif args.scenarios == "real":
        real = REAL_SCENARIOS
    elif args.scenarios == "all+real":
        synthetic = FULL_SCENARIOS
        real = REAL_SCENARIOS

    _print_header()
    results: List[ScenarioResult] = []
    for sc in synthetic:
        try:
            rs = run_scenario(reps=args.reps, warmup=args.warmup,
                              precisions=precisions, implementations=implementations,
                              **sc)
        except Exception as e:  # noqa: BLE001 - want to keep going on per-scenario errors
            print(f"{sc['name']:22s}  FAILED: {e}")
            continue
        for r in rs:
            results.append(r)
            _print_result(r)
    for sc in real:
        try:
            rs = _run_real_scenario(reps=args.reps, warmup=args.warmup,
                                    precisions=precisions, implementations=implementations,
                                    **sc)
        except Exception as e:  # noqa: BLE001
            print(f"{sc['name']:22s}  FAILED: {e}")
            continue
        if rs is None:
            print(f"{sc['name']:22s}  SKIPPED (dataset {sc['dataset']!r} not available)")
            continue
        for r in rs:
            results.append(r)
            _print_result(r)

    if results:
        # Group summary by (implementation, precision).
        for impl in implementations:
            for precision in precisions:
                subset = [r for r in results
                          if r.precision == precision and r.implementation == impl]
                if not subset:
                    continue
                avg_total = sum(r.total_speedup for r in subset) / len(subset)
                max_total = max(r.total_speedup for r in subset)
                meaningful_core = [r for r in subset if r.cuda_core_ms >= 0.05]
                tag = f"[{impl}/{precision}]"
                print()
                print(f"{tag} avg total speedup: {avg_total:.2f}x  (max {max_total:.2f}x)")
                if meaningful_core:
                    avg_core = sum(r.core_speedup for r in meaningful_core) / len(meaningful_core)
                    max_core = max(r.core_speedup for r in meaningful_core)
                    print(
                        f"{tag} avg core  speedup: {avg_core:.2f}x  (max {max_core:.2f}x, "
                        f"over {len(meaningful_core)}/{len(subset)} scenarios where cuda_core > 0.05ms)"
                    )

        # Direct kernel vs fused comparison (only where both are present).
        if "kernel" in implementations and "fused" in implementations:
            print()
            print("kernel vs fused (same precision, same scenario):")
            by_key: dict[Tuple[str, str], dict[str, ScenarioResult]] = {}
            for r in results:
                by_key.setdefault((r.scenario, r.precision), {})[r.implementation] = r
            ratios: List[float] = []
            for (scenario, precision), per in sorted(by_key.items()):
                if "kernel" in per and "fused" in per:
                    k = per["kernel"].cuda_total_ms
                    f = per["fused"].cuda_total_ms
                    ratio = k / max(f, 1e-6)
                    ratios.append(ratio)
                    print(f"  {scenario:38s} {precision:5s} kernel={k:6.2f}ms fused={f:6.2f}ms  fused_sp={ratio:5.2f}x")
            if ratios:
                avg = sum(ratios) / len(ratios)
                print(f"  fused vs kernel: avg {avg:.2f}x, max {max(ratios):.2f}x, min {min(ratios):.2f}x")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        print(f"\nWrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
