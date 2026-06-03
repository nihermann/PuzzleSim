from typing import Sequence, Dict, List, Union, Any

import torch
from torch import Tensor, nn
from torchvision import models as tv


class ConvBase(torch.nn.Module):
    """Base class for conv nets."""

    def __init__(self) -> None:
        super().__init__()
        self.slices = nn.ModuleList()

    def build(self, feature_per_block: Sequence[Sequence[int]], pretrained_features: nn.Sequential) -> None:
        """Build model from feature ranges."""
        for feature_range in feature_per_block:
            seq = torch.nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            self.slices.append(seq)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, n: Sequence[int]) -> Dict[int, Tensor]:
        """Process input."""
        output = {}

        for i in range(max(n) + 1):
            x = self.slices[i](x)
            if i in n:
                output[i] = x

        return output


class SqueezeNet(ConvBase):
    """SqueezeNet implementation."""

    def __init__(self) -> None:
        super().__init__()
        pretrained_features: nn.Sequential = tv.squeezenet1_1(weights=tv.SqueezeNet1_1_Weights.IMAGENET1K_V1).features

        feature_per_block = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 11), range(11, 12), range(12, 13)]

        self.build(feature_per_block, pretrained_features)


class AlexNet(ConvBase):
    """AlexNet implementation."""

    def __init__(self) -> None:
        super().__init__()
        alexnet_pretrained_features = tv.alexnet(weights=tv.AlexNet_Weights.IMAGENET1K_V1).features

        features_per_block = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 12)]

        self.build(features_per_block, alexnet_pretrained_features)


class Vgg16(ConvBase):
    """Vgg16 implementation."""

    def __init__(self) -> None:
        super().__init__()
        vgg_pretrained_features = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1).features

        features_per_block = [range(4), range(4, 9), range(9, 16), range(16, 23), range(23, 30)]

        self.build(features_per_block, vgg_pretrained_features)


class ScalingLayer(nn.Module):
    """Scaling layer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp: Tensor) -> Tensor:
        """Process input."""
        return (inp - self.shift) / self.scale
