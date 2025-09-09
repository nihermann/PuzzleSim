import re
from typing import List, Dict

import torch
from torch import Tensor, nn

from . import rr, rename_state_dict, RegexRule


VIT_RULES: List[RegexRule] = [
    rr(r'^embeddings\.patch_embeddings\.', 'patch_embed.proj.'),
    rr(r'^embeddings\.cls_token$', 'cls_token'),
    rr(r'^embeddings\.mask_token$', 'mask_token'),
    rr(r'^embeddings\.register_tokens$', 'storage_tokens'),
    rr(r'^layer\.(\d+)\.', r'blocks.\1.'),

    rr(r'\battention\.(q_proj|k_proj|v_proj)\b', r'attn.\1'),
    rr(r'\battention\.o_proj\b', 'attn.proj'),
    rr(r'\bmlp\.fc(\d+)\.(weight|bias)', r'mlp.w\1.\2'),

    rr(r'\bmlp\.gate_proj\b', 'mlp.w1'),
    rr(r'\bmlp\.up_proj\b', 'mlp.fc1'),
    rr(r'\bmlp\.down_proj\b', 'mlp.fc2'),

    rr(r'\blayer_scale1\.lambda1\b', 'ls1.gamma'),
    rr(r'\blayer_scale2\.lambda1\b', 'ls2.gamma'),
    # final norm keeps its name: norm.weight/bias
]

VIT_PLUS_RULES: List[RegexRule] = VIT_RULES + [
    rr(r'\bfc1\b', 'w2'),
    rr(r'\bfc2\b', 'w3'),
]

def fuse_qkv(mapped_sd: Dict[str, Tensor], author_sd: Dict[str, Tensor], depth: int) -> None:
    """
    Fuse separate q, k, v projections into a single qkv projection for each transformer block.
    This modifies mapped_sd in place.
    Args:
        mapped_sd: Dict[str, Tensor]: The state dict with separate q, k, v projections.
        author_sd: Dict[str, Tensor]: The author's state dict (for key reference).
        depth: int: The number of transformer blocks.

    Returns:
    """
    for i in range(depth):
        # author qkv targets
        Wq = mapped_sd.pop(f"blocks.{i}.attn.q_proj.weight")
        Wk = mapped_sd.pop(f"blocks.{i}.attn.k_proj.weight")
        Wv = mapped_sd.pop(f"blocks.{i}.attn.v_proj.weight")
        W = torch.cat([Wq, Wk, Wv], dim=0)
        qkv_key = f"blocks.{i}.attn.qkv.weight"
        assert qkv_key in author_sd, f"Expected {qkv_key} in author state dict"
        mapped_sd[qkv_key] = W

        # biases
        # there is no k bias
        bq = mapped_sd.pop(f"blocks.{i}.attn.q_proj.bias")
        bv = mapped_sd.pop(f"blocks.{i}.attn.v_proj.bias")
        bias_key = f"blocks.{i}.attn.qkv.bias"
        assert bias_key in author_sd, f"Expected {bias_key} in author state dict"
        mapped_sd[bias_key] = torch.cat([bq, torch.zeros_like(bq), bv], dim=0)
        mapped_sd[bias_key + "_mask"] = torch.zeros_like(mapped_sd[bias_key])

        for p in ("q_proj", "k_proj", "v_proj"):
            mapped_sd.pop(f"blocks.{i}.attn.{p}.weight", None)
            mapped_sd.pop(f"blocks.{i}.attn.{p}.bias", None)

def map_vit(huggingface_model: nn.Module, author_model: nn.Module, is_plus: bool = False) -> Dict[str, Tensor]:
    """
    Map a Huggingface ViT model state dict to the author's state dict format.
    Args:
        huggingface_model: nn.Module: The Huggingface ViT model.
        author_model: nn.Module: The author's ViT model (for shape reference).
        is_plus: bool: Whether the model is ViT-Plus variant.

    Returns:
        Dict[str, Tensor]: The mapped state dict.
    """
    # Plain renames
    rule_set = VIT_PLUS_RULES if is_plus else VIT_RULES
    mapped = rename_state_dict(huggingface_model.state_dict(), rule_set)

    # Fuse qkv (depth = infer from keys)
    author_sd = author_model.state_dict()
    depth = max((int(re.match(r"^blocks\.(\d+)\.", k).group(1))
                 for k in author_sd.keys() if k.startswith("blocks.")), default=-1) + 1
    fuse_qkv(mapped, author_sd, depth)
    mapped['mask_token'] = mapped['mask_token'].squeeze(dim=0)

    # the values of rope_embed.periods don't seem to matter,
    # and the huggingface model has different values than the author model anyway,
    # so we just hardcode the author's values here for compatibility
    mapped['rope_embed.periods'] = torch.tensor([
        1.0, 1.3359375, 1.78125, 2.375, 3.15625, 4.21875, 5.625,
        7.5, 10.0, 13.3125, 17.75, 23.75, 31.625, 42.25, 56.25, 75.0]
    ).to(mapped['mask_token'].device)
    return mapped
