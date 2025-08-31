# <img src="https://www.svgrepo.com/show/510149/puzzle-piece.svg" width="22"/> Puzzle Similarity

-----
by Nicolai Hermann, Jorge Condor, and Piotr Didyk  

<p align="left">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/cube.svg" width=13 height=13> <a target="_blank" href="https://nihermann.github.io/puzzlesim/index.html">Project Page</a>
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/file-pdf.svg" width=13 height=13> <a target="_blank" href="https://arxiv.org/abs/2411.17489">Paper</a>
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/database.svg" width=13 height=13> <a target="_blank" href="https://huggingface.co/datasets/nihermann/annotated-3DGS-artifacts">Data</a>
  <img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/file-circle-plus.svg" width=13 height=13> <a target="_blank" href="puzzlesim/data/Puzzle_Similarity_Supplemental.pdf">Supplemental</a>
</p>
<p align="left">
  <a target="_blank" href="https://arxiv.org/abs/2411.17489"><img src=https://img.shields.io/badge/arXiv-2411.17489-b31b1b.svg></a>
  <img src="https://github.com/nihermann/PuzzleSim/actions/workflows/tests.yml/badge.svg" alt="test results">
  <img src="https://img.shields.io/badge/license-Apache 2.0-green" alt="License">
</p>

This repository contains the implementation of the cross-reference metric PuzzleSim and a dedicated demo for the paper "Puzzle Similarity: A Perceptually-Guided Cross-Reference Metric for Artifact Detection in 3D Scene Reconstructions".

### Requirements
```shell
pip install -r requirements.txt
```
### Usage and Demo
Please find the demo in `demo.ipynb` to see how to run the metric on some example sets. In order to run the demo, you need to pull the data from another repository. Do this by either cloning the repository using
```shell
git clone https://github.com/nihermann/PuzzleSim.git --recursive
```
or if you already cloned the repository without the data submodule, you can download the submodule using
```shell
git submodule update --init --recursive
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
