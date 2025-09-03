from typing import List, Dict

from torch import Tensor, nn

from . import rr, rename_state_dict, RegexRule

CONV_RULES: List[RegexRule] = [
    rr(r'^stages\.(\d+)\.downsample_layers\.(\d+)\.', r'downsample_layers.\1.\2.'),
    rr(r'^stages\.(\d+)\.layers\.(\d+)\.', r'stages.\1.\2.'),
    rr(r'\bdepthwise_conv\b', 'dwconv'),
    rr(r'\blayer_norm\b', 'norm'),
    rr(r'\bpointwise_conv1\b', 'pwconv1'),
    rr(r'\bpointwise_conv2\b', 'pwconv2'),
    rr(r'^layer_norm\.', 'norm.'),
]

def map_conv(hugging_face_model: nn.Module, author_model: nn.Module) -> Dict[str, Tensor]:
    """
    Map a Huggingface ConvNeXt model state dict to the author's state dict format.
    Args:
        hugging_face_model: nn.Module: The Huggingface ConvNeXt model.
        author_model: nn.Module: The author's ConvNeXt model (for shape reference).

    Returns:

    """
    mapped = rename_state_dict(hugging_face_model.state_dict(), CONV_RULES)

    # Mirror final norm → norms.3.* if the author model expects it
    author_sd = author_model.state_dict()
    for suf in ("weight", "bias"):
        src = f"norm.{suf}"
        dst = f"norms.3.{suf}"
        if src in mapped and dst in author_sd and author_sd[dst].shape == mapped[src].shape:
            mapped[dst] = mapped[src]
    return mapped
