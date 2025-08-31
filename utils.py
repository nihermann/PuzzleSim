import os
from pathlib import Path
from typing import Tuple, List, Optional
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image

def load_images(dataset: str, device: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    base_path = Path("PuzzleSim-demo-data") / "samples" / dataset
    priors, _ = _read_images(base_path / "prior", device=device)
    test_images, names = _read_images(base_path / "test", device=device)

    return priors, test_images, names


def _read_images(directory: Path, device="cuda:0") -> Tuple[torch.Tensor, List[str]]:
    images = []
    image_names = []
    for fname in os.listdir(directory):
        img = Image.open(directory / fname)
        images.append(F.to_tensor(img).unsqueeze(0)[:, :3, :, :].to(device))
        image_names.append(fname)
    return torch.cat(images, dim=0), image_names


@torch.no_grad()
def plot_image_tensor(tensor: torch.Tensor) -> None:
    tensor = tensor.squeeze().permute(1, 2, 0).cpu()
    plt.imshow(tensor)
    plt.axis("off")
    plt.show()


@torch.no_grad()
def plot_heatmap_tensor(tensor: torch.Tensor) -> None:
    tensor = tensor.squeeze().cpu()
    plt.imshow(tensor, cmap=cm.jet.reversed())
    plt.axis("off")
    plt.show()


@torch.no_grad()
def plot_image_tensor_row(tensor: torch.Tensor, titles: List[str]) -> None:
    assert tensor.ndim == 4, "Expected 4D tensor"
    tensor = tensor.cpu().permute(0, 2, 3, 1)
    N = tensor.shape[0]
    fig, axs = plt.subplots(1, N, figsize=(N * 4, 4))
    for i, ax in enumerate(axs):
        ax.imshow(tensor[i])
        ax.axis("off")
        ax.set_title(titles[i])
    plt.show()
