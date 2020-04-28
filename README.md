# Hierarchichally Aggregated Residual Transformation for Image Restoration

This repository is an official PyTorch implementation of the paper "Hierarchichally Aggregated Residual Transformation for Image Restoration", which is submitted to BMVC2020.

## Dependency

Our code is tested on Ubuntu 16.04 environment (Python3.6, PyTorch0.4.1, CUDA8.0, cuDNN7.0) with NVIDIA Tesla P100 GPUs.

## Image super resolution

### Training

#### Preparing training datasets

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
2. Specify '--dir_data' based on the HR and LR images path.
