from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Literal, Optional, Tuple, Any, get_args

import torch
from torch import Tensor, nn
from torchvision import transforms
from transformers import AutoModel

from puzzle_sim.models import AlexNet, Vgg16, SqueezeNet, ScalingLayer
from puzzle_sim.helpers import resize_tensor
from puzzle_sim.state_dict_converter import transfer_data
from puzzle_sim.state_dict_converter.convnext_mapper import map_conv
from puzzle_sim.state_dict_converter.vit_mapper import map_vit

VGGAlexSqueezeType = Literal['vgg', 'alex', 'squeeze']
Dinov3Type = Literal[
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
    'vits16', 'vits16plus', 'vitb16', 'vitl16'
]
NetType = Union[VGGAlexSqueezeType, Dinov3Type]


class FeatureExtractor(ABC):
    """
    Abstract base class defining interface for feature extraction from image tensors.

    This class serves as a base interface for different feature extraction implementations,
    providing a common method signature for computing features from input tensors at
    specified network layers.
    """
    model: nn.Module

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


def get_feature_extractor(net_type: Union[FeatureExtractor, NetType], **kwargs: Any) -> FeatureExtractor:
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
        net = VGGAlexSqueezeAdapter(net_type=net_type)
        net.eval()
        return net
    elif net_type in get_args(Dinov3Type):
        verbose = kwargs.get('verbose', False)
        net = DinoV3Adapter(dino_type=net_type, verbose=verbose)
        net.eval()
        return net
    else:
        raise ValueError(f"Net type {net_type} unknown.")


class VGGAlexSqueezeAdapter(nn.Module, FeatureExtractor):
    def __init__(
            self,
            net_type: VGGAlexSqueezeType,
    ) -> None:
        """Initializes a perceptual loss torch.nn.Module.

        Args:
            net_type: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            resize: If input should be resized to this size
        """
        super().__init__()
        self.scaling_layer = ScalingLayer()

        self.model = {"vgg": Vgg16, "alex": AlexNet, "squeeze": SqueezeNet}[net_type]()

    def compute_features(self, img: Tensor, n: Union[int, Sequence[int]], normalize: bool = False) -> Dict[int, Tensor]:
        if isinstance(n, int):
            n = range(n)

        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            img = 2. * img - 1.

        # normalize input
        in0_input = self.scaling_layer(img)

        feats = self.model(in0_input, n)

        return feats


class DinoV3Adapter(nn.Module, FeatureExtractor):
    def __init__(self, dino_type: Dinov3Type, verbose: bool = False) -> None:
        super().__init__()

        # using torch.hub causes rate limit issues in some environments, which can be fixed by this workaround
        # https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        # instantiate model, bc weights can only be loaded via transformers or if downloaded manually
        # we pull the full model class so we can use get_intermediate_layers
        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov3",
            model=f"dinov3_{dino_type}",
            pretrained=False,
            verbose=verbose
        )
        # pull weights from huggingface model zoo
        self.hugging_face_model = AutoModel.from_pretrained(
            f"facebook/dinov3-{dino_type.replace('_', '-')}-pretrain-lvd1689m",
            device_map="auto"
        )

        if "conv" in dino_type:
            target_state_dict = map_conv(self.hugging_face_model, self.model)
        elif "vit" in dino_type:
            target_state_dict = map_vit(self.hugging_face_model, self.model, is_plus='plus' in dino_type)
        else:
            raise ValueError(f"Unknown architecture; net_type must be one of {get_args(Dinov3Type)}.")
        transfer_data(self.model, target_state_dict, verbose=verbose)

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
            features = self.model.get_intermediate_layers(tensor, n=max(n) + 1, reshape=True)

        output: Dict[int, Tensor] = {}
        for i in n:
            # print(features[i].shape[-2:])
            output[i] = features[i]
        return output
