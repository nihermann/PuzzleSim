# SPDX-License-Identifier: Apache-2.0
"""Gradient / differentiable-loss tests for both PuzzleSim CUDA modes.

PuzzleSim can be used as a perceptual loss on a *query image*: the backbone
weights and the reference distribution stay frozen, and gradients flow back to
the input. These tests verify that path end to end.

What is covered:

1. ``forward`` produces a graph (``requires_grad``) and ``backward`` populates a
   finite, non-zero ``img.grad`` for both ``PuzzleSimCUDA`` (fp32/tf32/fp16) and
   ``PuzzleSimCUDAFused``.
2. The fast inference path is unchanged: with no grad required (or under
   ``torch.no_grad``) the output is detached and the custom kernels are used.
3. A finite-difference check: the analytic gradient matches a central
   finite-difference estimate of the directional derivative (exact-FP32 cuDNN).
4. An Adam loop actually minimises a PuzzleSim-based loss.

The tensor-core ``sim_max`` kernels have no backward, so whenever the query
carries grad the metric falls back to an exact FP32 ``sim_max``. That is why the
gradient tests do not need tensor-core hardware: they exercise the FP32 path
regardless of the model's nominal ``precision``.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from .conftest import requires_cuda, requires_ext, requires_demo_data


def _build(kind: str, refs: torch.Tensor, precision: str):
    if kind == "kernel":
        from puzzle_sim_cuda import PuzzleSimCUDA

        return PuzzleSimCUDA(refs, net_type="squeeze", precision=precision)
    if kind == "fused":
        from puzzle_sim_cuda import PuzzleSimCUDAFused

        return PuzzleSimCUDAFused(refs, precision=precision)
    raise ValueError(kind)


@contextlib.contextmanager
def _exact_fp32_cudnn():
    """Force deterministic, non-TF32 cuDNN/cuBLAS so finite differences line up
    with the analytic gradient (TF32 convs would add ~1e-3 noise)."""
    prev_det = torch.backends.cudnn.deterministic
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_mm_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_det
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_mm_tf32


_KINDS_PRECISIONS = [
    ("kernel", "fp32"),
    ("kernel", "tf32"),
    ("kernel", "fp16"),
    ("fused", "tf32"),
]


# ---------------------------------------------------------------------------
# 1. The output is differentiable w.r.t. the query image
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("kind,precision", _KINDS_PRECISIONS)
def test_backward_populates_query_grad(kind, precision):
    refs = torch.rand(4, 3, 64, 64, device="cuda")
    model = _build(kind, refs, precision)

    img = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    out = model(img)
    assert out.requires_grad, f"{kind}/{precision}: output is detached"

    loss = -out.mean()
    loss.backward()

    assert img.grad is not None, f"{kind}/{precision}: no grad produced"
    assert torch.isfinite(img.grad).all(), f"{kind}/{precision}: non-finite grad"
    assert img.grad.abs().max() > 0, f"{kind}/{precision}: grad is identically zero"


@requires_cuda
@requires_ext
@pytest.mark.parametrize("kind,precision", _KINDS_PRECISIONS)
def test_references_and_weights_stay_frozen(kind, precision):
    """Only the query image should accumulate grad; the model is frozen."""
    refs = torch.rand(3, 3, 64, 64, device="cuda")
    model = _build(kind, refs, precision)

    img = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    (-model(img).mean()).backward()

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert not trainable, f"{kind}/{precision}: backbone params are not frozen"


# ---------------------------------------------------------------------------
# 2. The fast inference path is untouched
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("kind,precision", _KINDS_PRECISIONS)
def test_inference_output_is_detached(kind, precision):
    refs = torch.rand(3, 3, 64, 64, device="cuda")
    model = _build(kind, refs, precision)

    img = torch.rand(1, 3, 64, 64, device="cuda")  # no requires_grad
    out = model(img)
    assert not out.requires_grad, f"{kind}/{precision}: inference output carries grad"


@requires_cuda
@requires_ext
@pytest.mark.parametrize("kind,precision", _KINDS_PRECISIONS)
def test_no_grad_context_uses_fast_path(kind, precision):
    """Even a leaf that requires grad is detached under ``torch.no_grad``."""
    refs = torch.rand(3, 3, 64, 64, device="cuda")
    model = _build(kind, refs, precision)

    img = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    with torch.no_grad():
        out = model(img)
    assert not out.requires_grad, f"{kind}/{precision}: grad recorded under no_grad"


# ---------------------------------------------------------------------------
# 3. Finite-difference check of the analytic gradient
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("kind", ["kernel", "fused"])
def test_finite_difference_directional(kind):
    """Central finite difference of the directional derivative along the
    gradient direction must match the analytic gradient norm.

    Using ``v = g/||g||`` gives the strongest directional signal (best SNR for
    a finite-difference probe through a deep network)."""
    torch.manual_seed(0)
    refs = torch.rand(2, 3, 32, 32, device="cuda")
    model = _build(kind, refs, "fp32")

    def loss_value(x: torch.Tensor) -> float:
        # Fresh leaf -> differentiable (PyTorch-op) path, matching the analytic
        # forward exactly. We only need the scalar value here.
        xr = x.detach().requires_grad_(True)
        with torch.enable_grad():
            return float(model(xr, normalize=True).sum().item())

    with _exact_fp32_cudnn():
        base = torch.rand(1, 3, 32, 32, device="cuda")

        img = base.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = model(img, normalize=True).sum()
        loss.backward()
        g = img.grad
        assert g is not None
        gnorm = g.norm()
        assert gnorm > 1e-4

        v = g / gnorm
        eps = 2e-3
        f_plus = loss_value(base + eps * v)
        f_minus = loss_value(base - eps * v)
        dd_fd = (f_plus - f_minus) / (2 * eps)
        dd_analytic = float(gnorm.item())  # (g . v) == ||g||

    rel = abs(dd_fd - dd_analytic) / max(abs(dd_analytic), 1e-6)
    assert rel < 5e-2, (
        f"{kind}: directional FD {dd_fd:.4f} vs analytic {dd_analytic:.4f} "
        f"(rel err {rel:.3%})"
    )


# ---------------------------------------------------------------------------
# 4. PuzzleSim is minimisable as a loss
# ---------------------------------------------------------------------------

@requires_cuda
@requires_ext
@pytest.mark.parametrize("kind,precision", _KINDS_PRECISIONS)
def test_adam_reduces_loss(kind, precision):
    """Optimising an image to *increase* similarity reduces ``-sim.mean()``."""
    refs = torch.rand(4, 3, 64, 64, device="cuda")
    model = _build(kind, refs, precision)

    img = torch.rand(1, 3, 64, 64, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([img], lr=5e-2)

    losses = []
    for _ in range(40):
        opt.zero_grad()
        loss = -model(img).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            img.clamp_(0.0, 1.0)
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    # Final loss is clearly below the start, and the tail is below the head.
    assert losses[-1] < losses[0] - 1e-3, f"{kind}/{precision}: no progress {losses[0]:.4f}->{losses[-1]:.4f}"
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5, f"{kind}/{precision}: not trending down"


@requires_cuda
@requires_ext
@requires_demo_data("garden")
def test_adam_reduces_loss_real_data(garden_data):
    """Same optimisation sanity check on a real reference distribution."""
    from puzzle_sim_cuda import PuzzleSimCUDA

    assert garden_data is not None
    priors, tests, _ = garden_data
    model = PuzzleSimCUDA(priors, net_type="squeeze", resize=(128, 128), precision="fp32")

    img = tests[0:1].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([img], lr=2e-2)

    first = last = None
    for step in range(25):
        opt.zero_grad()
        loss = -model(img).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            img.clamp_(0.0, 1.0)
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
