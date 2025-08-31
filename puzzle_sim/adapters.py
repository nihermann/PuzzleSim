from abc import ABC, abstractmethod
from typing import Dict, Union, Sequence, Literal, Optional, Tuple

from torch import Tensor, nn

from puzzle_sim.models import AlexNet, Vgg16, SqueezeNet, ScalingLayer
from puzzle_sim.helpers import resize_tensor

class FeatureExtractor(ABC):
    @abstractmethod
    def compute_features(self, tensor: Tensor, n: Union[int, Sequence[int]], normalize: bool) -> Dict[int, Tensor]:
        pass


def get_feature_extractor(net_type: Union[FeatureExtractor, Literal["alex", "vgg", "squeeze"]], **kwargs) -> FeatureExtractor:
    if isinstance(net_type, FeatureExtractor):
        return net_type

    elif net_type in ["alex", "vgg", "squeeze"]:
        net = AlexVGGSqueezeAdapter(net_type=net_type, **kwargs)
        net.eval()
        return net
    else:
        raise ValueError(f"Net type {net_type} unknown.")


class AlexVGGSqueezeAdapter(nn.Module, FeatureExtractor):
    def __init__(
            self,
            net_type: Literal["alex", "vgg", "squeeze"] = "alex",
            resize: Optional[Union[int, Tuple[int]]] = None,
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
            img = 2 * img - 1

        # normalize input
        in0_input = self.scaling_layer(img)

        # resize input if needed
        if self.resize is not None:
            in0_input = resize_tensor(in0_input, size=self.resize, align_corners=True)

        feats = self.net.forward(in0_input, n)

        return feats