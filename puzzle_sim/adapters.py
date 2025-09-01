from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Literal, Optional, Tuple, Any, get_args

import torch
from torch import Tensor, nn
from torchvision import transforms

from puzzle_sim.models import AlexNet, Vgg16, SqueezeNet, ScalingLayer
from puzzle_sim.helpers import resize_tensor, upsample

VGGAlexSqueezeType = Literal['vgg', 'alex', 'squeeze']
Dinov3Type = Literal['convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'vits16', 'vits16plus', 'vitb16', 'vitl16']
NetType = Union[VGGAlexSqueezeType, Dinov3Type]


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

        self.model = torch.hub.load(
            repo_or_dir="../dinov3",
            model=f"dinov3_{net_type}",
            source="local",
            weights=f"puzzle_sim/dino_models/dinov3_{net_type}_pretrain_lvd1689m{'-8aa4cbdd' if net_type == 'vitl16' else ''}.pth"
        )
        print(f"Dinov3 {net_type} has n={self.model.n_blocks} blocks.")
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

        features = self.model.get_intermediate_layers(tensor, n=max(n)+1, reshape=True)

        output: Dict[int, Tensor] = {}
        for i in n:
            print(features[i].shape[-2:])
            output[i] = features[i]
        return output
