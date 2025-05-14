# <img src="https://www.svgrepo.com/show/510149/puzzle-piece.svg" width="22"/> Puzzle Similarity

-----

by Nicolai Hermann, Jorge Condor, and Piotr Didyk  
[[Project page](https://nihermann.github.io/puzzlesim/index.html)] [[arXiv](https://arxiv.org/abs/2411.17489)]

This repository contains the implementation of the No-Reference metric PuzzleSim and a dedicated demo for the paper "Puzzle Similarity: A Perceptually-guided No-Reference Metric for Artifact Detection in 3D Scene Reconstructions".

### Requirements
```shell
pip install -r requirements.txt
```
### Usage and Demo
Please find the demo in `demo.ipynb` to see how to run the metric on some example sets. In order to run the demo you need to pull the data from another repository. Do this by either cloning the repository using
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
@misc{hermann2024puzzlesim,
      title={Puzzle Similarity: A Perceptually-guided Cross-Reference Metric for Artifact Detection in 3D Scene Reconstructions},
      author={Nicolai Hermann and Jorge Condor and Piotr Didyk},
      year={2024},
      eprint={2411.17489},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17489},
}
```