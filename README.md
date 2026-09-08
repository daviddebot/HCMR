# WIP: Interpretable Hierarchical Concept Reasoning through Attention-Guided Graph Learning

> **Work in progress.** This is an early, minimal public reference implementation. It has not yet been extensively tested; use it for research and evaluation with appropriate care.

This repository contains a small reference implementation of H-CMR from *Interpretable Hierarchical Concept Reasoning through Attention-Guided Graph Learning* (Debot et al., 2025), together with an MNIST-Addition training example. The repository does not yet include all experiments, comparison baselines, plots, and datasets from the paper.

## Included

- Learned DAG orientation from node priorities.
- Three-valued rule memory (positive, negative, irrelevant literals).
- Attention-guided rule selection and hierarchical inference.
- MNIST-Addition, with two digit-concept groups and sum nodes.
- One train-and-save script that works on CPU and CUDA.

## Install

Create an isolated environment and install the pinned packages:

```bash
conda env create -f environment.yml
conda activate hcmr-public
```

For a CUDA-enabled PyTorch build, follow the platform-specific command on the [PyTorch installation page](https://pytorch.org/get-started/locally/) before installing the remaining dependencies.

## Smoke test

The first run downloads MNIST into `data/`:

```bash
python train_mnist_addition.py --epochs 1 --train-limit 512 --test-limit 256 --batch-size 64 --device cpu
```

For a longer run, omit the dataset limits and increase `--epochs`. Checkpoints and metrics are written under `runs/mnist-addition/`.

## Citation

If you use this code, please cite the following work:
```bibtex
@article{debot2025hcmr,
  title={Interpretable Hierarchical Concept Reasoning through Attention-Guided Graph Learning},
  author={Debot, David and Barbiero, Pietro and Dominici, Gabriele and Marra, Giuseppe},
  journal={arXiv preprint arXiv:2506.21102},
  year={2025}
}
```

Paper: https://arxiv.org/abs/2506.21102



## License

This code is released under the [Apache License 2.0](LICENSE).
