from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Literal, Optional, Tuple, Any, get_args
import re

import torch
from torch import Tensor, nn
from torchvision import transforms
from transformers import AutoModel

from puzzle_sim.models import AlexNet, Vgg16, SqueezeNet, ScalingLayer
from puzzle_sim.helpers import resize_tensor, upsample

VGGAlexSqueezeType = Literal['vgg', 'alex', 'squeeze']
Dinov3Type = Literal['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'vits16', 'vits16plus', 'vitb16', 'vitl16']
NetType = Union[VGGAlexSqueezeType, Dinov3Type]


def map_hf_to_author(hf_sd: dict, author_model: nn.Module) -> dict:
    rename_inside = {
        "depthwise_conv": "dwconv",
        "layer_norm": "norm",
        "pointwise_conv1": "pwconv1",
        "pointwise_conv2": "pwconv2",
    }

    mapped = {}

    for k, v in hf_sd.items():
        new_k = k

        # 1) Downsample layers: stages.S.downsample_layers.K.(weight|bias)
        m = re.match(r"^stages\.(\d+)\.downsample_layers\.(\d+)\.(.+)$", new_k)
        if m:
            s, kidx, tail = m.groups()
            new_k = f"downsample_layers.{s}.{kidx}.{tail}"
        else:
            # 2) Block layers: stages.S.layers.B.*
            new_k = re.sub(r"^stages\.(\d+)\.layers\.(\d+)\.", r"stages.\1.\2.", new_k)

        # 3) Submodule renames inside blocks
        parts = new_k.split(".")
        parts = [rename_inside.get(p, p) for p in parts]
        new_k = ".".join(parts)

        # 4) Final LN at the top-level
        if new_k.startswith("layer_norm."):
            new_k = new_k.replace("layer_norm.", "norm.", 1)

        mapped[new_k] = v

    # 5) If author has norms.3.*, mirror top norm.* there (shape-checked)
    author_sd = author_model.state_dict()
    for suffix in ["weight", "bias"]:
        src = f"norm.{suffix}"
        dst = f"norms.3.{suffix}"
        if src in mapped and dst in author_sd and author_sd[dst].shape == mapped[src].shape:
            mapped[dst] = mapped[src].clone()

    return mapped


def safe_load(model: nn.Module, new_sd: dict, verbose: bool = False) -> None:
    model_sd = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in new_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped.append((k, v.shape, model_sd.get(k, None).shape if k in model_sd else None))
    missing = sorted(set(model_sd.keys()) - set(loadable.keys()))
    unexpected = sorted(set(new_sd.keys()) - set(model_sd.keys()))

    if verbose:
        msg = {
            "num_loadable": len(loadable),
            "num_missing_in_new": len(missing),
            "num_unexpected_in_new": len(unexpected),
            "some_missing": missing[:10],
            "some_unexpected": unexpected[:10],
            "some_skipped_shape_mismatch": skipped[:10],
        }
        print(msg)
    assert len(missing) == 0, f"Missing keys in new state dict: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys in new state dict: {unexpected}"
    assert len(skipped) == 0, f"Skipped keys due to shape mismatch: {skipped}"
    model.load_state_dict(loadable, strict=False)


class FeatureExtractor(ABC):
    """
    Abstract base class defining interface for feature extraction from image tensors.

    This class serves as a base interface for different feature extraction implementations,
    providing a common method signature for computing features from input tensors at
    specified network layers.
    """
    @abstractmethod
    def compute_features(self, tensor: Tensor, n: Union[int, Sequence[int]], normalize: bool) -> Dict[int, Tensor]:
        """
        Computes features for a given image tensor (B, C, H, W) or (C, H, W).

        This abstract method processes the input tensor and computes intermediate features
        based on the specified layer/block indices. Optionally, normalization can be
        applied to the image tensor to match the backbones' domain.

        Args:
            tensor: Input tensor from which features are computed.
            n: Either specify the number of blocks to return the features of or a tuple of ints, specifying the indices of the blocks to extract features from.
            normalize: Indicates whether the input tensor should be normalized according to the backbone used. If normalize=True it is assumed, that the tensor is in [0, 1].

        Returns:
            A dictionary where the keys are the indices of the computed features
            and the values are tensors containing the corresponding features in (B, C_i, H_i, W_i).
        """
        pass


def get_feature_extractor(net_type: Union[FeatureExtractor, NetType], **kwargs: Dict[str, Any]) -> FeatureExtractor:
    """
    Factory function that creates and returns a feature extractor instance based on the specified network type.

    Args:
        net_type (NetType): Either an existing FeatureExtractor instance or a string specifying the type of network to use.
        **kwargs: Additional keyword arguments passed to the feature extractor constructor

    Returns:
        FeatureExtractor: An initialized feature extractor instance in evaluation mode
    """
    if isinstance(net_type, FeatureExtractor):
        return net_type

    if net_type in get_args(VGGAlexSqueezeType):
        net = VGGAlexSqueezeAdapter(net_type=net_type, **kwargs)
        net.eval()
        return net
    elif net_type in get_args(Dinov3Type):
        net = DinoV3Adapter(net_type=net_type)
        net.eval()
        return net
    else:
        raise ValueError(f"Net type {net_type} unknown.")


class VGGAlexSqueezeAdapter(nn.Module, FeatureExtractor):
    def __init__(
            self,
            net_type: VGGAlexSqueezeType,
            resize: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        """Initializes a perceptual loss torch.nn.Module.

        Args:
            net_type: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            resize: If input should be resized to this size
        """
        super().__init__()
        self.resize = resize
        self.scaling_layer = ScalingLayer()

        self.net = {"vgg": Vgg16, "alex": AlexNet, "squeeze": SqueezeNet}[net_type]()

    def compute_features(self, img: Tensor, n: Union[int, Sequence[int]], normalize: bool = False) -> Dict[int, Tensor]:
        if isinstance(n, int):
            n = list(range(n))

        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            img = 2. * img - 1.

        # normalize input
        in0_input = self.scaling_layer(img)

        # resize input if needed
        if self.resize is not None:
            in0_input = resize_tensor(in0_input, size=self.resize, align_corners=True)

        feats = self.net.forward(in0_input, n)

        return feats

class DinoV3Adapter(nn.Module, FeatureExtractor):
    def __init__(self, net_type: Dinov3Type) -> None:
        super().__init__()

        # instantiate model, bc weights can only be loaded via transformers or if downloaded manually
        # we pull the full model class so we can use get_intermediate_layers
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model=f"dinov3_{net_type}",
            source="github",
            pretrained=False,
            #weights=f"puzzle_sim/dino_models/dinov3_{net_type}_pretrain_lvd1689m{'-8aa4cbdd' if net_type == 'vitl16' else ''}.pth"
        )
        # pull weights from huggingface model zoo
        state_dict = AutoModel.from_pretrained(
            f"facebook/dinov3-{net_type.replace('_', '-')}-pretrain-lvd1689m",
            device_map="auto"
        ).state_dict()

        safe_load(self.model, map_hf_to_author(state_dict, self.model))

        self.transform = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    @torch.no_grad()
    def compute_features(self, tensor: Tensor, n: Union[int, Sequence[int]], normalize: bool) -> Dict[int, Tensor]:
        if isinstance(n, int):
            n = range(n)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(dim=0)

        if normalize:
            tensor = self.transform(tensor)

        with torch.inference_mode():
            features = self.model.get_intermediate_layers(tensor, n=max(n)+1, reshape=True)

        output: Dict[int, Tensor] = {}
        for i in n:
            print(features[i].shape[-2:])
            output[i] = features[i]
        return output
