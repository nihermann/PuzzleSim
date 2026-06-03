# SPDX-License-Identifier: Apache-2.0
"""High-level PuzzleSim implementation backed by CUDA.

Mirrors the API of ``puzzle_sim.PuzzleSim`` but:

* swaps in custom CUDA kernels for L2 normalization, bilinear upsample, and
  a fused tensor-core ``sim_max`` (GEMM + per-row max in a single kernel),
* picks the ``sim_max`` precision at construction time:
    - ``"tf32"`` (default): tensor-core kernel, ~1e-3 abs error vs FP32,
    - ``"fp16"``: tensor-core kernel with FP16 inputs / FP32 accumulator,
      ~5e-3 abs error, fastest,
    - ``"fp32"``: chunked ``torch.matmul + max`` fallback for strict-precision
      tests or older GPUs without tensor cores,
* packs reference features once at construction into the layout the
  contraction expects, so the per-query cost is just one backbone forward
  plus one ``sim_max`` per layer.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from puzzle_sim import Alexnet, ScalingLayer, SqueezeNet, Vgg16, _resize_tensor


# The compiled extension is built next to this package. Make sure its directory
# is importable so ``import puzzle_sim_cuda_ext`` resolves regardless of how the
# package was first imported (e.g. before a test ``conftest`` has extended
# ``sys.path``). Using the single top-level name everywhere is important: the
# pybind11 module is single-phase init and must not be loaded under two names.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    import puzzle_sim_cuda_ext as _ext  # type: ignore[import-not-found]
    CUDA_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via fallback path
    _ext = None
    CUDA_EXT_AVAILABLE = False


Precision = Literal["fp32", "tf32", "fp16"]
_VALID_PRECISIONS = ("fp32", "tf32", "fp16")


_NET_CHANNELS: Dict[str, List[int]] = {
    "squeeze": [64, 128, 256, 384, 384, 512, 512],
    "alex":    [64, 192, 384, 256, 256],
    "vgg":     [64, 128, 256, 512, 512],
    "vgg16":   [64, 128, 256, 512, 512],
}


def _make_backbone(net_type: str) -> nn.Module:
    if net_type in ("vgg", "vgg16"):
        return Vgg16(pretrained=True, requires_grad=False)
    if net_type == "alex":
        return Alexnet(pretrained=True, requires_grad=False)
    if net_type == "squeeze":
        return SqueezeNet(pretrained=True, requires_grad=False)
    raise ValueError(f"Unknown net_type: {net_type!r}")


def _grad_active(img: torch.Tensor) -> bool:
    """True when autograd should record ops for ``img``.

    The custom CUDA kernels have no backward, so the differentiable code paths
    only kick in when grad is enabled *and* the input is part of a graph that
    requires it (e.g. a learnable image used as a perceptual loss).
    """
    return (
        torch.is_grad_enabled()
        and isinstance(img, torch.Tensor)
        and img.requires_grad
    )


def _normalize_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Channel-wise L2 normalization.

    Uses the CUDA kernel when available and applicable; otherwise falls back to
    differentiable PyTorch ops (also used whenever ``features`` carries grad,
    since the kernel has no backward).
    """
    if (
        features.is_cuda
        and CUDA_EXT_AVAILABLE
        and features.dtype == torch.float32
        and not features.requires_grad
    ):
        return _ext.normalize_features(features.contiguous(), eps)
    norm = torch.sqrt(eps + features.pow(2).sum(dim=1, keepdim=True))
    return features / norm


def _pack_reference(refs: torch.Tensor) -> torch.Tensor:
    """Transpose+flatten ``(N, C, H, W)`` references to ``(C, N*H*W)``.

    Done once at construction so the per-call ``sim_max`` does not need to
    materialise a transposed view of refs every forward.
    """
    if refs.is_cuda and CUDA_EXT_AVAILABLE and refs.dtype == torch.float32:
        return _ext.pack_reference(refs.contiguous())
    N, C, H, W = refs.shape
    return refs.transpose(0, 1).contiguous().reshape(C, N * H * W)


# Peak transient bytes per matmul chunk when computing scores. 256 MB keeps us
# well under typical 6-8 GB free VRAM after the backbone is loaded, and the
# tile is large enough that cuBLAS hits its high-throughput kernels.
_SCORE_CHUNK_BYTES = 256 * 1024 * 1024


def _device_capability_at_least(major: int, minor: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (major, minor)


def _validate_precision(precision: str) -> Precision:
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {_VALID_PRECISIONS!r}, got {precision!r}"
        )
    return precision  # type: ignore[return-value]


def _sim_max(
    query_feats: torch.Tensor,
    refs_packed: torch.Tensor,
    query_hw: Tuple[int, int],
    precision: Precision = "fp32",
) -> torch.Tensor:
    """Per-pixel max similarity of ``query_feats`` against the packed refs.

    Dispatch:
      * ``precision == "tf32"`` / ``"fp16"``: tensor-core kernel from the C++
        extension. ``refs_packed`` must match: FP32 for tf32, FP16 for fp16.
      * ``precision == "fp32"``: chunked ``torch.matmul + max``. ``refs_packed``
        must be FP32.

    Args:
        query_feats: ``(1, C, qH, qW)`` or ``(C, qH, qW)`` query features
            (already L2 normalized).
        refs_packed: ``(C, N*rH*rW)`` reference features in the correct dtype.
        query_hw: ``(qH, qW)`` of the query.
        precision: one of ``"fp32"``, ``"tf32"``, ``"fp16"``.

    Returns:
        ``(qH, qW)`` similarity map (always FP32).
    """
    if query_feats.dim() == 4:
        query_feats = query_feats.squeeze(0)

    C, qH, qW = query_feats.shape
    qHW = qH * qW
    assert query_hw == (qH, qW)
    query_packed = query_feats.reshape(C, qHW).contiguous()

    if (
        precision in ("tf32", "fp16")
        and CUDA_EXT_AVAILABLE
        and query_packed.is_cuda
        and not query_packed.requires_grad
    ):
        if precision == "tf32":
            assert refs_packed.dtype == torch.float32, (
                f"refs_packed must be float32 for tf32 sim_max, got {refs_packed.dtype}"
            )
            return _ext.sim_max_tf32(query_packed, refs_packed, [qH, qW])
        # fp16
        assert refs_packed.dtype == torch.float16, (
            f"refs_packed must be float16 for fp16 sim_max, got {refs_packed.dtype}"
        )
        return _ext.sim_max_fp16(query_packed.half(), refs_packed, [qH, qW])

    # FP32 fallback via chunked matmul. Picks a K chunk such that the per-step
    # score tile (qHW * chunk) stays under _SCORE_CHUNK_BYTES.
    K = refs_packed.size(1)
    bytes_per_element = query_packed.element_size()
    max_chunk = max(1, _SCORE_CHUNK_BYTES // max(1, qHW * bytes_per_element))
    chunk = min(K, max_chunk)

    q_t = query_packed.transpose(0, 1)  # (qHW, C); non-contig view is fine, cuBLAS handles it.
    if chunk >= K:
        scores = q_t @ refs_packed                          # (qHW, K)
        sim_map = scores.max(dim=1).values
    else:
        sim_map = torch.full((qHW,), float("-inf"),
                             device=query_packed.device,
                             dtype=query_packed.dtype)
        for k0 in range(0, K, chunk):
            k1 = min(k0 + chunk, K)
            partial = q_t @ refs_packed[:, k0:k1]           # (qHW, chunk)
            torch.maximum(sim_map, partial.max(dim=1).values, out=sim_map)

    return sim_map.reshape(qH, qW)


def _bilinear_upsample(
    sim_map: torch.Tensor,
    out_hw: Tuple[int, int],
    align_corners: bool = True,
) -> torch.Tensor:
    """Bilinear upsample a 2D ``(H, W)`` similarity map to ``out_hw``."""
    if (
        sim_map.is_cuda
        and CUDA_EXT_AVAILABLE
        and sim_map.dtype == torch.float32
        and sim_map.dim() == 2
        and not sim_map.requires_grad
    ):
        return _ext.bilinear_upsample(sim_map.contiguous(), list(out_hw), align_corners)
    return F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0),
        size=out_hw,
        mode="bilinear",
        align_corners=align_corners,
    ).squeeze(0).squeeze(0)


class PuzzleSimCUDA(nn.Module):
    """CUDA-accelerated PuzzleSim metric.

    Compared to :class:`puzzle_sim.PuzzleSim`, this implementation:

    * Pre-packs reference features into the layout expected by the
      contraction, so this preparation happens once instead of on every
      forward.
    * Uses custom CUDA kernels for channel-wise L2 normalization, bilinear
      upsample, and (for ``precision != "fp32"``) the fused GEMM + max.
    * Picks the ``sim_max`` precision via the ``precision`` argument.

    The backbone CNN is still PyTorch's torchvision model in eval mode; that
    part of the pipeline is already cuDNN-backed.

    Args:
        reference: ``(N, C, H, W)`` tensor with the reference distribution.
        net_type: ``'squeeze'`` (default, as in the paper), ``'alex'`` or
            ``'vgg'``.
        resize: optional ``(H, W)`` to bilinearly resize inputs to before
            running the backbone.
        precision: one of ``"tf32"`` (default, sm_80+), ``"fp16"`` (sm_70+),
            or ``"fp32"`` (chunked cuBLAS GEMM; works everywhere).
    """

    def __init__(
        self,
        reference: torch.Tensor,
        net_type: Literal["alex", "vgg", "squeeze"] = "squeeze",
        resize: Optional[Tuple[int, int]] = None,
        precision: Precision = "tf32",
    ) -> None:
        super().__init__()
        if net_type not in _NET_CHANNELS:
            raise ValueError(f"Unknown net_type: {net_type!r}")

        precision = _validate_precision(precision)
        if precision == "tf32" and not _device_capability_at_least(8):
            # Tensor-core TF32 needs Ampere. Silently fall back to FP32.
            precision = "fp32"
        if precision == "fp16" and not _device_capability_at_least(7):
            precision = "fp32"
        if precision != "fp32" and not CUDA_EXT_AVAILABLE:
            precision = "fp32"

        self.net_type = net_type
        self.resize = resize
        self.precision: Precision = precision
        self.chns = _NET_CHANNELS[net_type]
        self.L = len(self.chns)
        self.device = reference.device

        self.net = _make_backbone(net_type).to(reference.device).eval()
        self.scaling_layer = ScalingLayer().to(reference.device)

        for p in self.parameters():
            p.requires_grad = False

        self.reference = reference
        # Cached, packed reference features. Filled lazily on first forward so
        # we never run a 4090 backbone forward inside __init__ when the user
        # might still be moving things around. ``_packed_refs`` always stores
        # FP32 versions; ``_packed_refs_fp16`` is populated on demand.
        self._packed_refs: Optional[Dict[int, torch.Tensor]] = None
        self._packed_refs_fp16: Dict[int, torch.Tensor] = {}
        self._ref_shapes: Optional[Dict[int, Tuple[int, int]]] = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _scaled_input(self, img: torch.Tensor, normalize: bool) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if normalize:
            img = 2 * img - 1
        img = self.scaling_layer(img)
        if self.resize is not None:
            img = _resize_tensor(img, size=self.resize, align_corners=True)
        return img

    def compute_features(
        self,
        img: torch.Tensor,
        normalize: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """Return normalized backbone features for ``img``.

        Gradients flow back to ``img`` when it requires grad and grad is enabled
        (the backbone weights stay frozen). Otherwise the call runs under
        ``no_grad`` and uses the fast custom kernels, exactly as before.
        """
        img = img.to(self.device)
        ctx = contextlib.nullcontext() if _grad_active(img) else torch.no_grad()
        with ctx:
            scaled = self._scaled_input(img, normalize)
            outs = self.net.forward(scaled)
            return {k: _normalize_features(outs[k]) for k in range(self.L)}

    def _ensure_reference_packed(self) -> None:
        if self._packed_refs is not None:
            return
        with torch.no_grad():
            ref_feats = self.compute_features(self.reference, normalize=True)
        self._packed_refs = {}
        self._ref_shapes = {}
        for k, feats in ref_feats.items():
            self._packed_refs[k] = _pack_reference(feats)
            self._ref_shapes[k] = (feats.shape[2], feats.shape[3])

    def _refs_for_layer(self, layer: int) -> torch.Tensor:
        """Return packed refs for ``layer`` in the dtype matching ``self.precision``."""
        assert self._packed_refs is not None
        if self.precision == "fp16":
            cached = self._packed_refs_fp16.get(layer)
            if cached is None:
                cached = self._packed_refs[layer].half().contiguous()
                self._packed_refs_fp16[layer] = cached
            return cached
        return self._packed_refs[layer]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        img: torch.Tensor,
        layers: Tuple[int, ...] = (2, 3, 4),
        normalize: bool = True,
        reduction: Literal["mean", "sum"] = "sum",
        weights: Optional[Tuple[float, ...]] = (0.67, 0.2, 0.13),
    ) -> torch.Tensor:
        """Return a similarity map for ``img`` against the reference distribution."""
        if weights is not None and len(weights) != len(layers):
            raise ValueError("Number of weights must match number of layers.")

        self._ensure_reference_packed()
        assert self._packed_refs is not None and self._ref_shapes is not None

        if img.dim() == 3:
            img = img.unsqueeze(0)
        out_hw = img.shape[-2:]

        # When the query carries grad, compute sim_max in exact FP32 (the
        # tensor-core kernels have no backward); references stay frozen.
        differentiable = _grad_active(img)
        query_feats = self.compute_features(img, normalize=normalize)

        sims: List[torch.Tensor] = []
        for i, layer in enumerate(layers):
            q = query_feats[layer]
            qH, qW = q.shape[2], q.shape[3]
            if differentiable:
                r = self._packed_refs[layer]
                sim_map = _sim_max(q, r, (qH, qW), precision="fp32")
            else:
                r = self._refs_for_layer(layer)
                sim_map = _sim_max(q, r, (qH, qW), precision=self.precision)
            if weights is not None:
                sim_map = sim_map * weights[i]
            sims.append(_bilinear_upsample(sim_map, out_hw, align_corners=True))

        result = torch.stack(sims, dim=0).sum(dim=0)
        if reduction == "mean":
            result = result / len(sims)
        return result


def PuzzleSim(  # noqa: N802 - keep parity with original API
    reference: torch.Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "squeeze",
    resize: Optional[Tuple[int, int]] = None,
    use_cuda: bool = True,
    precision: Precision = "tf32",
    implementation: Literal["kernel", "fused"] = "kernel",
):
    """Drop-in replacement for :class:`puzzle_sim.PuzzleSim`.

    When ``use_cuda`` is true *and* the CUDA extension is built *and* the
    reference tensor is on a CUDA device, returns a :class:`PuzzleSimCUDA`
    instance (``implementation='kernel'``) or :class:`PuzzleSimCUDAFused`
    (``implementation='fused'``). Otherwise falls back to the original PyTorch
    implementation.

    The fused implementation is currently only available for
    ``net_type='squeeze'``; for other backbones it silently falls back to the
    kernel implementation.
    """
    if (
        use_cuda
        and CUDA_EXT_AVAILABLE
        and torch.cuda.is_available()
        and reference.is_cuda
    ):
        if implementation == "fused" and net_type == "squeeze":
            from .puzzle_sim_cuda_fused import PuzzleSimCUDAFused
            return PuzzleSimCUDAFused(reference, resize=resize, precision=precision)
        return PuzzleSimCUDA(reference, net_type=net_type, resize=resize, precision=precision)
    from puzzle_sim import PuzzleSim as _RefPuzzleSim
    return _RefPuzzleSim(reference, net_type=net_type, resize=resize)


__all__ = ["PuzzleSimCUDA", "PuzzleSim", "CUDA_EXT_AVAILABLE", "Precision"]
