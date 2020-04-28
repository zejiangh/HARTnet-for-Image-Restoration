# Hierarchichally Aggregated Residual Transformation for Image Restoration

This repository is an official PyTorch implementation of the paper "Hierarchichally Aggregated Residual Transformation for Image Restoration", which is submitted to BMVC2020.

## Dependency

Our code is tested on Ubuntu 16.04 environment (Python3.6, PyTorch0.4.1, CUDA8.0, cuDNN7.0) with NVIDIA Tesla P100 GPUs.

## Image super resolution

### Training

#### Preparing training datasets

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
2. Specify '--dir_data' based on the HR and LR images path.

#### Training from scratch

```Shell
# Super-resolution scaling factor x2, x3, x4

# scale x2, input=48x48, output=96x96
python main.py --model DDHRN --scale 2 --patch_size 96  --save DDHRN_scalex2 --reset --epochs 800 --n_GPUs 2 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_range '1-800/801-810'

# scale x3, input=48x48, output=144x144
python main.py --model DDHRN --scale 3 --patch_size 144  --save DDHRN_scalex3 --reset --epochs 800 --n_GPUs 2 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_range '1-800/801-810'

# scale x4, input=48x48, output=192x192
python main.py --model DDHRN --scale 4 --patch_size 192  --save DDHRN_scalex4 --reset --epochs 800 --n_GPUs 2 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_range '1-800/801-810'
```
