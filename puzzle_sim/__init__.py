from typing import List, Optional, Tuple, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from puzzle_sim.adapters import FeatureExtractor, get_feature_extractor
from puzzle_sim.helpers import upsample


def find_best_matching_piece(refs: Tensor, img: Tensor, stride: Optional[int] = 4, mem_save: bool = True) -> Tensor:
    """
    Find the best matching piece in refs for each spatial location in img.
    Args:
        refs: Tensor of shape (N, C, H, W) representing the reference distribution
        img: Tensor of shape (C, H, W) or (1, C, H, W) representing the input image
        stride: int, controls how many slices are in one block. If memory is a concern decrease this number. Depending on the hardware and application different values can be optimal.
        mem_save: bool, should generally be set to True. In rare cases, when the image resolution and the number of reference images is small the naive implementation can be faster.

    Returns:
        Tensor: the similarity map of the input image to the reference distribution of shape (H, W).
    """
    if mem_save and stride is None:
        raise ValueError("stride must not be none when mem_save=True.")

    if img.ndim == 3:
        img = img.unsqueeze(dim=0)

    refs = F.normalize(refs, p=2, dim=1)
    img = F.normalize(img, p=2, dim=1).squeeze()

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

class PuzzleSim(nn.Module):
    def __init__(self, reference: Tensor, net_type: Literal["alex", "vgg", "squeeze"] = "squeeze", resize: Optional[Tuple[int, int]]=None) -> None:
        """
        Instantiates the PuzzleSim metric on a given reference distribution.
        Find the paper at https://arxiv.org/abs/2411.17489
        Args:
            reference: tensor of shape (N, C, H, W) representing the reference distribution
            net_type: which base network to use, choose between ['alex','vgg','squeeze']. Defaults to 'squeeze' as detailed in our paper.
            resize: tuple to resize the references and inputs to. Recommended if the image sizes change or are too large.
        """
        super().__init__()
        self.feature_extractor: FeatureExtractor = get_feature_extractor(net_type, resize=resize)
        self.to(reference.device)
        self.reference = reference
        self.reference_feats = None


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

        feats = self.feature_extractor.compute_features(img, layers, normalize)
        if self.reference_feats is None or any(layer not in self.reference_feats for layer in layers):
            self.reference_feats = self.feature_extractor.compute_features(self.reference, layers, normalize)

        sims = []
        for i, layer in enumerate(layers):
            sim_map = find_best_matching_piece(self.reference_feats[layer], feats[layer].squeeze(), stride, mem_save)

            if weights is not None:
                sim_map = sim_map * weights[i]

            sims.append(upsample(sim_map, out_hw=img.shape[-2:], align_corners=True).squeeze())

        sim_summary: Tensor = sum(sims)  # type: ignore[assignment]
        if reduction == 'mean':
            return sim_summary / len(sims)

        return sim_summary






