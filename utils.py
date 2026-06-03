import os
from pathlib import Path
from typing import Tuple, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image


def load_images(dataset: str, device: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    base_path = Path("PuzzleSim-demo-data") / "samples" / dataset
    priors, _ = _read_images(base_path / "prior", device=device)
    test_images, names = _read_images(base_path / "test", device=device)

    return priors, test_images, names


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.tensor(arr, dtype=torch.float32)


def _read_images(directory: Path, device="cuda:0") -> Tuple[torch.Tensor, List[str]]:
    images = []
    image_names = []
    for fname in os.listdir(directory):
        img = Image.open(directory / fname)
        # F.to_tensor uses torch.from_numpy under the hood, which can fail with some NumPy/PyTorch combos.
        # Replace with our own safe conversion that uses torch.tensor (copies) instead of torch.from_numpy (views).
        images.append(_pil_to_tensor(img).unsqueeze(0).to(device))
        image_names.append(fname)
    return torch.cat(images, dim=0), image_names


@torch.no_grad()
def plot_image_tensor(tensor: torch.Tensor) -> None:
    numpy = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(numpy)
    plt.axis("off")
    plt.show()


@torch.no_grad()
def plot_heatmap_tensor(tensor: torch.Tensor) -> None:
    numpy = tensor.squeeze().cpu().numpy()
    plt.imshow(numpy, cmap=cm.jet.reversed())
    plt.axis("off")
    plt.show()


@torch.no_grad()
def plot_image_tensor_row(tensor: torch.Tensor, titles: List[str]) -> None:
    assert tensor.ndim == 4, "Expected 4D tensor"
    numpy = tensor.cpu().permute(0, 2, 3, 1).numpy()
    N = numpy.shape[0]
    fig, axs = plt.subplots(1, N, figsize=(N * 4, 4))
    for i, ax in enumerate(axs):
        ax.imshow(numpy[i])
        ax.axis("off")
        ax.set_title(titles[i])
    plt.show()
