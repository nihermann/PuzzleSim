from typing import NamedTuple
import torch
from torch import nn

from torchmetrics.utilities.imports import _TORCHVISION_GREATER_EQUAL_0_13


_weight_map = {
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    "alexnet": "AlexNet_Weights",
    "vgg16": "VGG16_Weights",
}

def _get_net(net: str, pretrained: bool) -> nn.modules.container.Sequential:
    """Get torchvision network.

    Args:
        net: Name of network
        pretrained: If pretrained weights should be used

    """
    from torchvision import models as tv
    if _TORCHVISION_GREATER_EQUAL_0_13:
        if pretrained:
            pretrained_features = getattr(tv, net)(weights=getattr(tv, _weight_map[net]).IMAGENET1K_V1).features
        else:
            pretrained_features = getattr(tv, net)(weights=None).features
    else:
        pretrained_features = getattr(tv, net)(pretrained=pretrained).features
    return pretrained_features


class SqueezeNet(torch.nn.Module):
    """SqueezeNet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        pretrained_features = _get_net("squeezenet1_1", pretrained)

        self.N_slices = 7
        slices = []
        feature_ranges = [range(2), range(2, 5), range(5, 8), range(8, 10), range(10, 11), range(11, 12), range(12, 13)]
        for feature_range in feature_ranges:
            seq = torch.nn.Sequential()
            for i in feature_range:
                seq.add_module(str(i), pretrained_features[i])
            slices.append(seq)

        self.slices = nn.ModuleList(slices)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """Process input."""

        class _SqueezeOutput(NamedTuple):
            relu1: torch.Tensor
            relu2: torch.Tensor
            relu3: torch.Tensor
            relu4: torch.Tensor
            relu5: torch.Tensor
            relu6: torch.Tensor
            relu7: torch.Tensor

        relus = []
        for slice_ in self.slices:
            x = slice_(x)
            relus.append(x)
        return _SqueezeOutput(*relus)


class Alexnet(torch.nn.Module):
    """Alexnet implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        alexnet_pretrained_features = _get_net("alexnet", pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h

        class _AlexnetOutputs(NamedTuple):
            relu1: torch.Tensor
            relu2: torch.Tensor
            relu3: torch.Tensor
            relu4: torch.Tensor
            relu5: torch.Tensor

        return _AlexnetOutputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class Vgg16(torch.nn.Module):
    """Vgg16 implementation."""

    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        vgg_pretrained_features = _get_net("vgg16", pretrained)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> NamedTuple:
        """Process input."""
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        class _VGGOutputs(NamedTuple):
            relu1_2: torch.Tensor
            relu2_2: torch.Tensor
            relu3_3: torch.Tensor
            relu4_3: torch.Tensor
            relu5_3: torch.Tensor

        return _VGGOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


class ScalingLayer(nn.Module):
    """Scaling layer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Process input."""
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()

        layers = [nn.Dropout()] if use_dropout else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),  # type: ignore[list-item]
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input."""
        return self.model(x)
