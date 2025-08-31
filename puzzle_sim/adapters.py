from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Literal, Optional, Tuple

import torch
from torch import Tensor, nn


from puzzle_sim.models import Alexnet, Vgg16, SqueezeNet, NetLinLayer, ScalingLayer
from puzzle_sim.helpers import resize_tensor

class FeatureExtractor(ABC):
    @abstractmethod
    def compute_features(self, tensor: Tensor, n: Union[int, Sequence[int]], normalize: bool) -> Dict[int, Tensor]:
        pass


def get_feature_extractor(net_type: Union[FeatureExtractor, Literal["alex", "vgg", "squeeze"]] = "alex", **kwargs) -> FeatureExtractor:
    if isinstance(net_type, FeatureExtractor):
        return net_type

    elif net_type in ["alex", "vgg", "squeeze"]:
        net = AlexVGGSqueezeAdapter(net=net_type, **kwargs)
        net.eval()
        return net
    else:
        raise ValueError(f"Net type {net_type} unknown.")


class AlexVGGSqueezeAdapter(nn.Module, FeatureExtractor):
    def __init__(
            self,
            net: Literal["alex", "vgg", "squeeze"] = "alex",
            spatial: bool = False,
            pnet_tune: bool = False,
            use_dropout: bool = False,
            resize: Optional[Union[int, Tuple[int]]] = None,
    ) -> None:
        """Initializes a perceptual loss torch.nn.Module.

        Args:
            net: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            spatial: If input should be spatial averaged
            pnet_tune: If backprop should be enabled for both backbone and linear layers
            use_dropout: If dropout layers should be added
            resize: If input should be resized to this size
        """
        super().__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.spatial = spatial
        self.resize = resize
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = Vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = Alexnet  # type: ignore[assignment]
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = SqueezeNet  # type: ignore[assignment]
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        else:
            raise ValueError(f"Net type {self.pnet_type} unknown.")
        self.L = len(self.chns)

        self.net: Union[Vgg16, Alexnet, SqueezeNet] = net_type(requires_grad=self.pnet_tune)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        if self.pnet_type == "squeeze":  # 7 layers for squeezenet
            self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
            self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
            self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)  # type: ignore[assignment]

        for param in self.parameters():
            param.requires_grad = False

    def compute_features(self, img: Tensor, n: Union[int, Sequence[int]], normalize: bool = False) -> Dict[int, Tensor]:
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            img = 2 * img - 1

        # normalize input
        in0_input = self.scaling_layer(img)

        # resize input if needed
        if self.resize is not None:
            in0_input = resize_tensor(in0_input, size=self.resize, align_corners=True)

        outs = self.net.forward(in0_input)

        feats: Dict[int, torch.Tensor] = {}
        for l in range(self.L):
            feats[l] = outs[l]

        return feats