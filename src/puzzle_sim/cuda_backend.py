from __future__ import annotations

import os
from typing import Dict, Literal, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor

Precision = Literal["auto", "fp32", "tf32", "fp16"]
ResolvedPrecision = Literal["fp32", "tf32", "fp16"]
Backend = Literal["auto", "torch", "cuda"]

_VALID_BACKENDS = ("auto", "torch", "cuda")
_VALID_PRECISIONS = ("auto", "fp32", "tf32", "fp16")
_RESOLVED_PRECISIONS = ("fp32", "tf32", "fp16")
_DEFAULT_PRECISION: ResolvedPrecision = "fp32"
_SCORE_CHUNK_BYTES = int(os.environ.get("PUZZLE_SIM_CUDA_SCORE_CHUNK_MB", "256")) * 1024 * 1024

try:
    from . import _cuda_ext as _ext  # type: ignore[attr-defined]

    CUDA_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional binary extension
    _ext = None
    CUDA_EXT_AVAILABLE = False


def is_available() -> bool:
    """Return True when the optional compiled CUDA extension is importable."""
    return CUDA_EXT_AVAILABLE


def validate_backend(backend: Backend) -> None:
    """Validate a backend selector."""
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS!r}, got {backend!r}.")


def should_use_packed_cuda(refs: Tensor, query: Tensor) -> bool:
    """True when the packed CUDA/PyTorch matmul backend can safely run."""
    return (
        refs.is_cuda
        and query.is_cuda
        and refs.dtype == torch.float32
        and query.dtype == torch.float32
        and not refs.requires_grad
    )


def grad_active(*tensors: Tensor) -> bool:
    """True when autograd should record ops for at least one tensor."""
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def _device_capability_at_least(device: torch.device, major: int, minor: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.cuda.get_device_capability(device_index) >= (major, minor)


def resolve_precision(precision: Precision = "auto", device: Optional[torch.device] = None) -> ResolvedPrecision:
    """Resolve user/env precision to a supported mode.

    The default is exact FP32 to preserve the public ``puzzle_sim.PuzzleSim``
    numerics. Users who want the inference-only tensor-core kernels can opt in
    with ``cuda_precision="tf32"``/``"fp16"`` or ``PUZZLE_SIM_CUDA_PRECISION``.
    """
    if precision not in _VALID_PRECISIONS:
        raise ValueError(f"cuda_precision must be one of {_VALID_PRECISIONS!r}, got {precision!r}")

    env_precision = os.environ.get("PUZZLE_SIM_CUDA_PRECISION")
    if precision == "auto":
        precision = cast(Precision, env_precision or _DEFAULT_PRECISION)
    if precision not in _RESOLVED_PRECISIONS:
        raise ValueError(f"cuda_precision must be one of {_VALID_PRECISIONS!r}, got {precision!r}")

    resolved = cast(ResolvedPrecision, precision)
    if resolved != "fp32" and not CUDA_EXT_AVAILABLE:
        return "fp32"
    if device is None:
        device = torch.device("cuda")
    if resolved == "tf32" and not _device_capability_at_least(device, 8):
        return "fp32"
    if resolved == "fp16" and not _device_capability_at_least(device, 7):
        return "fp32"
    return resolved


def normalize_features(features: Tensor, eps: float = 1e-8) -> Tensor:
    """Channel-wise L2 normalization with optional CUDA-kernel acceleration."""
    if (
        CUDA_EXT_AVAILABLE
        and features.is_cuda
        and features.dtype == torch.float32
        and not features.requires_grad
    ):
        return _ext.normalize_features(features.contiguous(), eps)
    norm = torch.sqrt(eps + features.pow(2).sum(dim=1, keepdim=True))
    return features / norm


def pack_reference(refs: Tensor) -> Tensor:
    """Pack ``(N, C, H, W)`` features as ``(C, N*H*W)`` for the contraction."""
    if CUDA_EXT_AVAILABLE and refs.is_cuda and refs.dtype == torch.float32 and not refs.requires_grad:
        return _ext.pack_reference(refs.contiguous())
    n, c, h, w = refs.shape
    return refs.transpose(0, 1).contiguous().reshape(c, n * h * w)


def refs_for_precision(refs_packed_fp32: Tensor, precision: ResolvedPrecision) -> Tensor:
    if precision == "fp16":
        return refs_packed_fp32.half().contiguous()
    return refs_packed_fp32


class PackedReferenceCache:
    """Cache packed reference features for the integrated CUDA backend."""

    def __init__(self, precision: Precision = "auto") -> None:
        self.precision = precision
        self._packed_refs: Dict[int, Tensor] = {}
        self._packed_refs_fp16: Dict[int, Tensor] = {}

    def clear(self) -> None:
        """Clear all cached packed tensors."""
        self._packed_refs.clear()
        self._packed_refs_fp16.clear()

    def similarity(self, layer: int, refs: Tensor, query: Tensor) -> Tensor:
        """Return the query-vs-reference similarity map for one feature layer."""
        query_requires_grad = grad_active(query)
        precision = "fp32" if query_requires_grad else resolve_precision(self.precision, query.device)
        refs_packed = self._refs_for_layer(layer, refs, precision)
        query_norm = normalize_features(query)
        return sim_max(query_norm, refs_packed, precision=precision)

    def _refs_for_layer(self, layer: int, refs: Tensor, precision: ResolvedPrecision) -> Tensor:
        refs_packed = self._packed_refs.get(layer)
        if refs_packed is None:
            refs_norm = normalize_features(refs)
            refs_packed = pack_reference(refs_norm)
            self._packed_refs[layer] = refs_packed
            self._packed_refs_fp16.pop(layer, None)

        if precision != "fp16":
            return refs_packed

        refs_packed_fp16 = self._packed_refs_fp16.get(layer)
        if refs_packed_fp16 is None:
            refs_packed_fp16 = refs_packed.half().contiguous()
            self._packed_refs_fp16[layer] = refs_packed_fp16
        return refs_packed_fp16


def sim_max(
    query_feats: Tensor,
    refs_packed: Tensor,
    *,
    precision: ResolvedPrecision = "fp32",
) -> Tensor:
    """Per-pixel max similarity against packed reference features."""
    if query_feats.dim() == 4:
        if query_feats.size(0) != 1:
            raise ValueError("The CUDA backend expects a single query image.")
        query_feats = query_feats.squeeze(0)
    if query_feats.dim() != 3:
        raise ValueError(f"query_feats must be 3D or single-image 4D, got shape {tuple(query_feats.shape)}")

    c, qh, qw = query_feats.shape
    qhw = qh * qw
    query_packed = query_feats.reshape(c, qhw).contiguous()
    query_requires_grad = query_packed.requires_grad

    if (
        precision in ("tf32", "fp16")
        and CUDA_EXT_AVAILABLE
        and query_packed.is_cuda
        and not query_requires_grad
    ):
        if precision == "tf32":
            if refs_packed.dtype != torch.float32:
                raise TypeError(f"refs_packed must be float32 for tf32, got {refs_packed.dtype}")
            return _ext.sim_max_tf32(query_packed, refs_packed, [qh, qw])
        if refs_packed.dtype != torch.float16:
            raise TypeError(f"refs_packed must be float16 for fp16, got {refs_packed.dtype}")
        return _ext.sim_max_fp16(query_packed.half(), refs_packed, [qh, qw])

    if refs_packed.dtype != query_packed.dtype:
        refs_packed = refs_packed.to(dtype=query_packed.dtype)

    k = refs_packed.size(1)
    bytes_per_element = query_packed.element_size()
    max_chunk = max(1, _SCORE_CHUNK_BYTES // max(1, qhw * bytes_per_element))
    chunk = min(k, max_chunk)

    query_t = query_packed.transpose(0, 1)
    if chunk >= k:
        scores = query_t @ refs_packed
        return scores.max(dim=1).values.reshape(qh, qw)

    sim_map = torch.full(
        (qhw,),
        float("-inf"),
        device=query_packed.device,
        dtype=query_packed.dtype,
    )
    for k0 in range(0, k, chunk):
        k1 = min(k0 + chunk, k)
        partial = query_t @ refs_packed[:, k0:k1]
        partial_max = partial.max(dim=1).values
        if query_requires_grad:
            sim_map = torch.maximum(sim_map, partial_max)
        else:
            torch.maximum(sim_map, partial_max, out=sim_map)
    return sim_map.reshape(qh, qw)


def bilinear_upsample(sim_map: Tensor, out_hw: Tuple[int, int], align_corners: bool = True) -> Tensor:
    """Upsample a 2D similarity map with optional CUDA-kernel acceleration."""
    if (
        CUDA_EXT_AVAILABLE
        and sim_map.is_cuda
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


def find_best_matching_piece(
    refs: Tensor,
    img: Tensor,
    *,
    precision: Precision = "auto",
) -> Tensor:
    """CUDA/PyTorch packed implementation of PuzzleSim's spatial max."""
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4 or img.size(0) != 1:
        raise ValueError("img must have shape (C, H, W) or (1, C, H, W)")
    if refs.ndim != 4:
        raise ValueError("refs must have shape (N, C, H, W)")

    query_requires_grad = grad_active(img)
    resolved_precision = "fp32" if query_requires_grad else resolve_precision(precision, img.device)
    refs_norm = normalize_features(refs)
    img_norm = normalize_features(img)
    refs_packed = pack_reference(refs_norm)
    refs_packed = refs_for_precision(refs_packed, resolved_precision)
    return sim_max(img_norm, refs_packed, precision=resolved_precision)


def get_version_info() -> Dict[str, object]:
    """Return runtime information for diagnostics."""
    info: Dict[str, object] = {
        "cuda_extension_available": CUDA_EXT_AVAILABLE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    return info


__all__ = [
    "Backend",
    "CUDA_EXT_AVAILABLE",
    "Precision",
    "ResolvedPrecision",
    "PackedReferenceCache",
    "bilinear_upsample",
    "find_best_matching_piece",
    "get_version_info",
    "is_available",
    "normalize_features",
    "pack_reference",
    "refs_for_precision",
    "resolve_precision",
    "should_use_packed_cuda",
    "sim_max",
    "validate_backend",
]
