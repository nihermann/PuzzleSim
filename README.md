# <img src="https://www.svgrepo.com/show/510149/puzzle-piece.svg" width="22"/> Puzzle Similarity

<p align="left">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/cube.svg" width=13 height=13> <a target="_blank" href="https://nihermann.github.io/puzzlesim/index.html">Project Page</a>
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/file-pdf.svg" width=13 height=13> <a target="_blank" href="https://arxiv.org/abs/2411.17489">Paper</a>
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/database.svg" width=13 height=13> <a target="_blank" href="https://huggingface.co/datasets/nihermann/annotated-3DGS-artifacts">Data</a>
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/file-circle-plus.svg" width=13 height=13> <a target="_blank" href="https://nihermann.github.io/puzzlesim/data/Puzzle_Similarity_Supplemental.pdf">Supplemental</a>
</p>

by Nicolai Hermann, Jorge Condor, and Piotr Didyk  

-----


<p align="left">
  <a target="_blank" href="https://arxiv.org/abs/2411.17489"><img src=https://img.shields.io/badge/arXiv-2411.17489-b31b1b.svg></a>
  <img src="https://github.com/nihermann/PuzzleSim/actions/workflows/tests.yml/badge.svg" alt="test results">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

This repository contains the implementation of the cross-reference metric PuzzleSim and a dedicated demo for the paper "Puzzle Similarity: A Perceptually-Guided Cross-Reference Metric for Artifact Detection in 3D Scene Reconstructions".

### News
- (04-09-2025) Dinov3 🦖 backbones (ConvNeXt & Vits) are now supported
- (31-08-2025) Major refactoring allowing to easily add custom backbones and automated cross-platform testing
- (25-06-2025) PuzzleSim was officially accepted to ICCV 2025 in Hawaii 🌸!
- (29-11-2024) Official code release


### Requirements
To use the metric please install it locally as a package. The package requires Python 3.8 or higher. If you wish to use dinov3 backbones you must have Python 3.10 or higher and `transformers>=4.56`:
```shell
pip install -e .
```


### Usage
You can use the metric in your own code as follows:
```python
from puzzle_sim import PuzzleSim

priors = ...  # load priors from file with shape (N, C, H, W) in [0, 1]
test_image = ...  # load test image (C, H, W) or (1, C, H, W) in [0, 1]
puzzle = PuzzleSim(reference=priors, net_type='squeeze')

similarity_map = puzzle(test_image)  # (H, W) similarity map in [0, 1]
```
To use dinov3 backbones you must be logged in to HuggingFace (`hf auth login`), requested access to the models on HuggingFace and review the necessary requirements above. You can request access to the models [here](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009).
In code, you have to adapt the `puzzle()` call as default arguments assume the configuration from the paper. We have not tested optimal weights for dinov3 backbones yet, so we recommend to use a simple average over all layers (which has shown similar performance to the configurations in the paper):
```python
from puzzle_sim import PuzzleSim

priors = ...  # load priors from file with shape (N, C, H, W) in [0, 1]
test_image = ...  # load test image (C, H, W) or (1, C, H, W) in [0, 1]
puzzle = PuzzleSim(reference=priors, net_type='convnext_tiny')

similarity_map = puzzle(test_image, layers=range(5), weights=None, reduction='mean')  # (H, W) similarity map in [0, 1]
```
> If your GPU runs out of memory, try reducing the `stride` parameter in the forward call, this will reduce memory consumption. On the other hand, with small image dimensions the naive implementation might be faster although requiring much more memory (set `mem_save=False`).

### Demo
Please find the demo in `demo.ipynb` to see how to run the metric on some example sets. In order to run the demo, you need to pull the data from another repository. Do this by either cloning the repository using
```shell
git clone https://github.com/nihermann/PuzzleSim.git --recursive
```
or if you already cloned the repository without the data submodule, you can download the submodule using
```shell
git submodule update --init --recursive
```

### Add Your Own Backbones
You can extend PuzzleSim with your own backbone models. To get started, inherit from `adapters.FeatureExtractor` and implement the `compute_features` method.

There are two ways to use your backbone:
1. Directly in the constructor:  
```python
PuzzleSim(..., net_type=YourBackbone())
```
2. Via the factory function: Register your backbone in `adapters.get_feature_extractor` and add the corresponding string to `adapters.net_type`, so you can refer to it by that string: 
```python
PuzzleSim(..., 'your_backbone')
```

#### 💡 Contributing
If you’d like to share your backbone with the community, feel free to open a pull request. Please make sure that:
- [ ] Your backbone is publicly available (e.g., on HuggingFace or PyTorch Hub)
- [ ] you’ve registered it in the factory function `adapters.get_feature_extractor`,
- [ ] extended `adapters.net_type` (so the tests pick it up automatically),
- [ ] and all tests pass (run `pytest` in the project root).

For development, we recommend installing the package in editable mode with dev requirements:
```shell
pip install -e .[dev]
```

### Citation
If you find this work useful, please consider citing:
```bibtex
@inproceedings{hermann2025puzzlesim,
      title={Puzzle Similarity: A Perceptually-Guided Cross-Reference Metric for Artifact Detection in 3D Scene Reconstructions},
      author={Nicolai Hermann and Jorge Condor and Piotr Didyk},
      booktitle={ICCV},
      year={2025},
}
```
