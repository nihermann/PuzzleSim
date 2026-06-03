# SPDX-License-Identifier: Apache-2.0
"""Fully fused C++ implementation of PuzzleSim over SqueezeNet1.1.

Unlike :class:`puzzle_sim_cuda.PuzzleSimCUDA`, which orchestrates the per-layer
backbone / normalize / sim-max / upsample chain in Python, this class hands
the entire pipeline to a single C++ entrypoint. The C++ side owns:

* the SqueezeNet weights (pre-extracted from torchvision once at construction),
* the cuDNN handle that backs ``at::cudnn_convolution_relu`` (the fused
  conv + bias + ReLU kernel),
* the packed, cached reference features per layer,
* the tensor-core ``sim_max`` kernels from this extension.

The only work that stays in Python is the (cheap) scaling layer + optional
input resize, plus turning the dict of weights into the order the C++ class
expects. Per forward, there is exactly one round-trip into C++.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from puzzle_sim import ScalingLayer, SqueezeNet, _resize_tensor

# Reuse the exact module object the kernel path already imported (it also makes
# sure the package directory is on sys.path). Sharing the single instance avoids
# loading the single-phase pybind11 extension under two different names.
from .puzzle_sim_cuda import _ext, _grad_active


_HAVE_FUSED = _ext is not None and hasattr(_ext, "FusedSqueezeNet")


Precision = Literal["fp32", "tf32", "fp16"]
_VALID_PRECISIONS = ("fp32", "tf32", "fp16")


def _device_capability_at_least(major: int, minor: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (major, minor)


def _extract_squeeze_weights(net: SqueezeNet) -> List[torch.Tensor]:
    """Flatten the SqueezeNet1.1 backbone weights into the 50-tensor list
    the C++ ``FusedSqueezeNet`` constructor expects.

    Layout (see ``src/fused_inference.cpp``):
        [0]    conv0.weight  (64, 3, 3, 3)
        [1]    conv0.bias    (64,)
        [2..49] eight Fire modules, each contributing six tensors in order
                squeeze.weight, squeeze.bias,
                expand1x1.weight, expand1x1.bias,
                expand3x3.weight, expand3x3.bias.
    """
    weights: List[torch.Tensor] = []
    weights.append(net.slices[0][0].weight.detach().contiguous())
    weights.append(net.slices[0][0].bias.detach().contiguous())
    fires_seen = 0
    for s_idx in range(1, 7):
        for m in net.slices[s_idx]:
            if hasattr(m, "squeeze") and hasattr(m, "expand1x1") and hasattr(m, "expand3x3"):
                weights.append(m.squeeze.weight.detach().contiguous())
                weights.append(m.squeeze.bias.detach().contiguous())
                weights.append(m.expand1x1.weight.detach().contiguous())
                weights.append(m.expand1x1.bias.detach().contiguous())
                weights.append(m.expand3x3.weight.detach().contiguous())
                weights.append(m.expand3x3.bias.detach().contiguous())
                fires_seen += 1
    assert fires_seen == 8, f"Expected 8 Fire modules in SqueezeNet1.1, got {fires_seen}"
    assert len(weights) == 50, f"Expected 50 weight tensors, got {len(weights)}"
    return weights


class PuzzleSimCUDAFused(nn.Module):
    """End-to-end fused PuzzleSim over SqueezeNet1.1.

    Numerically equivalent to :class:`puzzle_sim_cuda.PuzzleSimCUDA` with
    ``net_type='squeeze'`` at the same ``precision``, but it bundles the
    backbone forward, normalize, sim-max and upsample into one C++ call.

    Currently only the SqueezeNet backbone is supported; AlexNet/VGG will
    follow the same pattern.

    Args:
        reference: ``(N, 3, H, W)`` reference distribution.
        resize: optional ``(H, W)`` target. Applied as in the reference
            ``puzzle_sim`` implementation (``area`` mode when downscaling,
            anti-aliased bilinear otherwise).
        precision: one of ``"tf32"`` (default, sm_80+), ``"fp16"`` (sm_70+),
            ``"fp32"``. Bound to the ``sim_max`` step; the cuDNN convolutions
            silently use TF32 internally when ``torch.backends.cudnn.allow_tf32``
            is true (matches the kernel-path behaviour).

    Gradients: the fused inference path bypasses autograd, so when a forward is
    called with a query that requires grad (e.g. optimising an image against the
    reference distribution) it transparently delegates to a kernel-mode model
    with exact FP32 ``sim_max``. ``loss.backward()`` then populates the query's
    ``.grad``; the backbone weights and references stay frozen.
    """

    def __init__(
        self,
        reference: torch.Tensor,
        resize: Optional[Tuple[int, int]] = None,
        precision: Precision = "tf32",
    ) -> None:
        super().__init__()
        if not _HAVE_FUSED:
            raise RuntimeError(
                "PuzzleSimCUDAFused requires the puzzle_sim_cuda_ext CUDA extension "
                "with the fused inference module built in. Rebuild the extension."
            )
        if not reference.is_cuda:
            raise RuntimeError(
                "PuzzleSimCUDAFused requires reference to live on a CUDA device."
            )
        if precision not in _VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {_VALID_PRECISIONS!r}, got {precision!r}")
        if precision == "tf32" and not _device_capability_at_least(8):
            precision = "fp32"
        if precision == "fp16" and not _device_capability_at_least(7):
            precision = "fp32"

        self.resize = resize
        self.precision: Precision = precision
        self.device = reference.device

        backbone = SqueezeNet(pretrained=True, requires_grad=False).to(reference.device).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        weights = _extract_squeeze_weights(backbone)
        self._backbone = backbone  # kept alive so weight tensors stay valid

        self.scaling_layer = ScalingLayer().to(reference.device)

        self._fused = _ext.FusedSqueezeNet(weights, precision)
        self.reference = reference
        self._ref_layers: Optional[Tuple[int, ...]] = None
        # Lazily built when a forward needs gradients (see _grad_forward).
        self._grad_model: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # Input preprocessing kept in Python; the heavy lifting lives in C++.
    # ------------------------------------------------------------------
    def _scaled_input(self, img: torch.Tensor, normalize: bool) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if normalize:
            img = 2 * img - 1
        img = self.scaling_layer(img)
        if self.resize is not None:
            img = _resize_tensor(img, size=self.resize, align_corners=True)
        return img.contiguous()

    def _ensure_reference_packed(self, layers: Tuple[int, ...]) -> None:
        if self._ref_layers is not None and tuple(self._ref_layers) == tuple(layers):
            return
        with torch.no_grad():
            scaled_refs = self._scaled_input(self.reference, normalize=True)
            self._fused.set_reference(scaled_refs, list(layers))
        self._ref_layers = tuple(layers)

    # Public for parity with PuzzleSimCUDA tests + benchmarks.
    def compute_features(
        self,
        img: torch.Tensor,
        normalize: bool = False,
        layers: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
    ) -> Dict[int, torch.Tensor]:
        img = img.to(self.device)
        with torch.no_grad():
            scaled = self._scaled_input(img, normalize)
            return self._fused.compute_features(scaled, list(layers))

    def _grad_forward(
        self,
        img: torch.Tensor,
        layers: Tuple[int, ...],
        normalize: bool,
        reduction: Literal["mean", "sum"],
        weights: Optional[Tuple[float, ...]],
    ) -> torch.Tensor:
        """Differentiable forward for query-image gradients.

        The fused C++ path has no backward (it bypasses autograd entirely), so
        when ``img`` carries grad we route through a kernel-mode model built
        from the same pretrained weights and reference. That path computes the
        backbone with torchvision (frozen weights, grad w.r.t. the input) and an
        exact FP32 ``sim_max``, so ``loss.backward()`` populates ``img.grad``.
        """
        if self._grad_model is None:
            from .puzzle_sim_cuda import PuzzleSimCUDA

            self._grad_model = PuzzleSimCUDA(
                self.reference,
                net_type="squeeze",
                resize=self.resize,
                precision="fp32",
            )
        return self._grad_model(
            img,
            layers=layers,
            normalize=normalize,
            reduction=reduction,
            weights=weights,
        )

    def forward(
        self,
        img: torch.Tensor,
        layers: Tuple[int, ...] = (2, 3, 4),
        normalize: bool = True,
        reduction: Literal["mean", "sum"] = "sum",
        weights: Optional[Tuple[float, ...]] = (0.67, 0.2, 0.13),
    ) -> torch.Tensor:
        if weights is not None and len(weights) != len(layers):
            raise ValueError("Number of weights must match number of layers.")

        if _grad_active(img):
            return self._grad_forward(img, layers, normalize, reduction, weights)

        self._ensure_reference_packed(tuple(layers))

        if img.dim() == 3:
            img = img.unsqueeze(0)
        out_hw = (int(img.shape[-2]), int(img.shape[-1]))

        with torch.no_grad():
            scaled = self._scaled_input(img.to(self.device), normalize)
            w_list = [float(w) for w in weights] if weights is not None else []
            return self._fused.forward(
                scaled,
                list(layers),
                w_list,
                reduction,
                list(out_hw),
            )


__all__ = ["PuzzleSimCUDAFused", "Precision"]
