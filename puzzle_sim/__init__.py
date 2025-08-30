# The authors of this paper have released the code under the Apache 2.0 open source license and extended the code to include the PuzzleSim metric.

# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Content copied from
# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# and
# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/pretrained_networks.py
# and with adjustments from
# https://github.com/richzhang/PerceptualSimilarity/pull/114/files
# due to package no longer being maintained
# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# All rights reserved.
# License under BSD 2-clause
import os
from typing import List, NamedTuple, Optional, Tuple, Union, Dict

import torch
from torch import Tensor, nn
import torch.functional as F
from typing_extensions import Literal

from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13

_weight_map = {
    "squeezenet1_1": "SqueezeNet1_1_Weights",
    "alexnet": "AlexNet_Weights",
    "vgg16": "VGG16_Weights",
}

if not _TORCHVISION_AVAILABLE:
    __doctest_skip__ = ["learned_perceptual_image_patch_similarity"]


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

            # file gets saved to cache directory and not the path that is later expected so we move it there
            model_file_name = os.path.basename(getattr(tv, _weight_map[net]).IMAGENET1K_V1.url)
            path_src = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", model_file_name)
            net_name = {
                "squeezenet1_1": "squeeze",
                "alexnet": "alex",
                "vgg16": "vgg"
            }[net]
            path_dest = os.path.join(os.path.dirname(__file__), "lpips_models", f"{net_name}.pth")
            os.makedirs(os.path.dirname(path_dest), exist_ok=True)
            if not os.path.exists(path_dest):
                os.rename(path_src, path_dest)
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

    def forward(self, x: Tensor) -> NamedTuple:
        """Process input."""

        class _SqueezeOutput(NamedTuple):
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor
            relu6: Tensor
            relu7: Tensor

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

    def forward(self, x: Tensor) -> NamedTuple:
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
            relu1: Tensor
            relu2: Tensor
            relu3: Tensor
            relu4: Tensor
            relu5: Tensor

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

    def forward(self, x: Tensor) -> NamedTuple:
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
            relu1_2: Tensor
            relu2_2: Tensor
            relu3_3: Tensor
            relu4_3: Tensor
            relu5_3: Tensor

        return _VGGOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _spatial_average(in_tens: Tensor, keep_dim: bool = True) -> Tensor:
    """Spatial averaging over height and width of images."""
    return in_tens.mean([2, 3], keepdim=keep_dim)


def _upsample(in_tens: Tensor, out_hw: Tuple[int, ...] = (64, 64), align_corners: bool = False) -> Tensor:
    """Upsample input with bilinear interpolation."""
    while len(in_tens.shape) < 4:
        in_tens = in_tens.unsqueeze(0)
    return nn.Upsample(size=out_hw, mode="bilinear", align_corners=align_corners)(in_tens)


def _normalize_tensor(in_feat: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize input tensor."""
    norm_factor = torch.sqrt(eps + torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / norm_factor


def _resize_tensor(x: Tensor, size: int = 64, align_corners: bool = False) -> Tensor:
    """https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/sample_similarity_lpips.py#L127C22-L132."""
    if isinstance(size, int) and x.shape[-1] > size and x.shape[-2] > size:
        return torch.nn.functional.interpolate(x, (size, size), mode="area")
    return torch.nn.functional.interpolate(x, (size, size) if isinstance(size, int) else size, mode="bilinear", align_corners=align_corners, antialias=True)


class ScalingLayer(nn.Module):
    """Scaling layer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None], persistent=False)
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None], persistent=False)

    def forward(self, inp: Tensor) -> Tensor:
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

    def forward(self, x: Tensor) -> Tensor:
        """Process input."""
        return self.model(x)


class _LPIPS(nn.Module):
    def __init__(
            self,
            pretrained: bool = True,
            net: Literal["alex", "vgg", "squeeze"] = "alex",
            spatial: bool = False,
            pnet_rand: bool = False,
            pnet_tune: bool = False,
            use_dropout: bool = True,
            model_path: Optional[str] = None,
            eval_mode: bool = True,
            resize: Optional[Union[int, Tuple[int]]] = None,
    ) -> None:
        """Initializes a perceptual loss torch.nn.Module.

        Args:
            pretrained: This flag controls the linear layers should be pretrained version or random
            net: Indicate backbone to use, choose between ['alex','vgg','squeeze']
            spatial: If input should be spatial averaged
            pnet_rand: If backbone should be random or use imagenet pre-trained weights
            pnet_tune: If backprop should be enabled for both backbone and linear layers
            use_dropout: If dropout layers should be added
            model_path: Model path to load pretained models from
            eval_mode: If network should be in evaluation mode
            resize: If input should be resized to this size

        """
        super().__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
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
        self.L = len(self.chns)

        self.net: Union[Vgg16, Alexnet, SqueezeNet] = net_type(pretrained=not self.pnet_rand, requires_grad=self.pnet_tune)

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

        # if pretrained:
        #     if model_path is None:
        #         model_path = os.path.abspath(
        #             os.path.join(inspect.getfile(self.__init__), "..", f"lpips_models/{net}.pth")  # type: ignore[misc]
        #         )
        #
        #     self.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)

        if eval_mode:
            self.eval()

        if not self.pnet_tune:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
            self, in0: Tensor, in1: Tensor, retperlayer: bool = False, normalize: bool = False, raw_outs: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # normalize input
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)

        # resize input if needed
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)
            in1_input = _resize_tensor(in1_input, size=self.resize)

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = _normalize_tensor(outs0[kk]), _normalize_tensor(outs1[kk])
            if not raw_outs:
                diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if raw_outs:
            return feats0, feats1

        res = []
        for kk in range(self.L):
            if self.spatial:
                res.append(_upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:])))
            else:
                res.append(_spatial_average(self.lins[kk](diffs[kk]), keep_dim=True))

        val: Tensor = sum(res)  # type: ignore[assignment]
        if retperlayer:
            return (val, res)
        return val


class _NoTrainLpips(_LPIPS):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainLpips":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


def _valid_img(img: Tensor, normalize: bool) -> bool:
    """Check that input is a valid image to the network."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check  # type: ignore[return-value]


def _lpips_update(img1: Tensor, img2: Tensor, net: nn.Module, normalize: bool) -> Tuple[Tensor, Union[int, Tensor]]:
    if not (_valid_img(img1, normalize) and _valid_img(img2, normalize)):
        raise ValueError(
            "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
            f" Got input with shape {img1.shape} and {img2.shape} and values in range"
            f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
            f" expected to be in the {[0, 1] if normalize else [-1, 1]} range."
        )
    loss = net(img1, img2, normalize=normalize).squeeze()
    return loss, img1.shape[0]


def _lpips_compute(sum_scores: Tensor, total: Union[Tensor, int], reduction: Literal["sum", "mean"] = "mean") -> Tensor:
    return sum_scores / total if reduction == "mean" else sum_scores


def learned_perceptual_image_patch_similarity(
        img1: Tensor,
        img2: Tensor,
        net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        reduction: Literal["sum", "mean", "none"] = "mean",
        normalize: bool = False,
        spatial: bool = False,
        resize: Optional[Union[int, Tuple[int]]] = None,
) -> Tensor:
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates perceptual similarity between two images.

    LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
    This measure has been shown to match human perception well. A low LPIPS score means that image patches are
    perceptual similar.

    Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
    chosen backbone (see `net_type` arg).

    Args:
        img1: first set of images
        img2: second set of images
        net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
        reduction: str indicating how to reduce over the batch dimension. Choose between `'sum'` or `'mean'`.
        normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
            to ``True`` will instead expect input to be in the ``[0,1]`` range.
        spatial: if spatial map should be returned or not

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(123)
        >>> from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
        >>> img1 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> img2 = (torch.rand(10, 3, 100, 100) * 2) - 1
        >>> learned_perceptual_image_patch_similarity(img1, img2, net_type='squeeze')
        tensor(0.1008)

    """
    net = _NoTrainLpips(net=net_type, spatial=spatial, resize=resize).to(device=img1.device, dtype=img1.dtype)
    loss, total = _lpips_update(img1, img2, net, normalize)
    if reduction != 'none':
        return _lpips_compute(loss.sum(), total, reduction)
    return loss


def find_best_matching_piece(refs: torch.Tensor, img: torch.Tensor, stride: Optional[int] = 4, mem_save: bool = True) -> torch.Tensor:
    if mem_save and stride is None:
        raise ValueError("stride must not be none when mem_save=True.")

    if img.ndim == 3:
        img = img.unsqueeze(dim=0)

    refs = _normalize_tensor(refs)
    img = _normalize_tensor(img).squeeze()

    if mem_save:
        N, C, H, W = refs.shape

        candidates = []
        # factor over h, the dimension that you max over
        for h in range(0, H, stride):
            sim = torch.einsum('cHW,ncwh->nHWwh', img, refs[:, :, h:h + stride, :])
            c_WH = (
                sim
                .reshape(N, H * W, -1)  # what was rows in sim is now last dimension
                .max(dim=-1)  # distribute max over ref.W
                .values  # get max values instead of indices
                .max(dim=0)  # distribute max over ref.N
                .values  # get max values instead of indices
            )
            candidates.append(c_WH)

        sim_map = (
            torch.stack(candidates, dim=0)
            .max(dim=0)  # distribute max over ref.H
            .values
            .view(H, W)  # reshape to spatial map
        )
    else:
        # flatten spatial dimensions
        h, w = img.shape[1:]
        flat_img = img.flatten(1, 2)
        flat_ref = refs.flatten(2, 3)

        # compute cosine similarity through outer product
        # [Ground Truth] x [Test]^T -> [N, H*W, H*W]
        # [N, C, H*W] x [C, H*W] -> [N, H*W, H*W]
        sim = torch.einsum('ncx,cy->nxy', flat_ref, flat_img)
        max_sim = sim.max(dim=1)[0]

        sim_map = (
            max_sim
            .unflatten(1, (h, w))
            .max(dim=0)[0]
        )

    return sim_map

class PuzzleSim(_NoTrainLpips):
    def __init__(self, reference: Tensor, net_type: Literal["alex", "vgg", "squeeze"] = "squeeze", resize: Optional[Tuple[int, int]]=None) -> None:
        """
        Instantiates the PuzzleSim metric on a given reference distribution.
        Find the paper at https://arxiv.org/abs/2411.17489
        Args:
            reference: tensor of shape (N, C, H, W) representing the reference distribution
            net_type: which base network to use, choose between ['alex','vgg','squeeze']. Defaults to 'squeeze' as detailed in our paper.
            resize: tuple to resize the references and inputs to. Recommended if the image sizes change or are too large.
        """
        super().__init__(
            net=net_type,
            resize=resize,
            pretrained=True,
            spatial=True,
            use_dropout=True
        )
        self.to(reference.device)
        self.reference = reference
        self.reference_feats = None

    def compute_features(self, img: Tensor, normalize: bool = False) -> Dict[int, Tensor]:
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            img = 2 * img - 1

        # normalize input
        in0_input = self.scaling_layer(img)

        # resize input if needed
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize, align_corners=True)

        outs = self.net.forward(in0_input)

        feats: Dict[int, torch.Tensor] = {}
        for l in range(self.L):
            feats[l] = outs[l]

        return feats

    def forward(self, img: Tensor, layers: List[int] = (2, 3, 4), normalize: bool = True, reduction: Literal['mean', 'sum'] = 'sum', weights: List[float] = (0.67, 0.2, 0.13), mem_save: bool = True, stride=4) -> Tensor:
        """
        Compute the PuzzleSim metric for an input image.
        Args:
            img: tensor of shape (1, C, H, W) or (C, H, W) representing the input image with values in the range [0, 1] (default) or [-1, 1] if normalize=False.
            layers: which layers of the network to use.
            normalize: set to True if input is in the [0, 1] range, False if input is in the [-1, 1] range.
            reduction: how to combine the similarity maps from different layers. Choose between 'mean' and 'sum'.
            weights: weights to apply to each layer's similarity map.
            mem_save: should generally be set to True (default). In rare cases, when the image resolution and the number of reference images is small the naive implementation can be faster.
            stride: controls how many slices are in one block. If memory is a concern decrease this number. Depending on the hardware and application different values can be optimal.

        Returns:
            Tensor: the similarity map of the input image to the reference distribution of shape (H, W).
        """
        if weights is not None and len(weights) != len(layers):
            raise ValueError("Number of weights must match number of layers.")

        feats = self.compute_features(img, normalize)
        if self.reference_feats is None:
            self.reference_feats = self.compute_features(self.reference, normalize)

        sims = []
        for i, layer in enumerate(layers):
            sim_map = find_best_matching_piece(self.reference_feats[layer], feats[layer].squeeze(), stride, mem_save)

            if weights is not None:
                sim_map = sim_map * weights[i]

            sims.append(_upsample(sim_map, out_hw=img.shape[-2:], align_corners=True).squeeze())

        sim_summary: Tensor = sum(sims)  # type: ignore[assignment]
        if reduction == 'mean':
            return sim_summary / len(sims)

        return sim_summary






