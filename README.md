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

### Testing

#### Preparing benchmark datasets

1. You can evaluate our models on widely-used benchmark datasets, including Set5, Set14, BSD100, and Urban100. Download benchmark datasets from [here](https://cv.snu.ac.kr/research/EDSR/benchmark.tar).
2. Set '--dir_data' to where benchmark folder is located in order to evaluate PSNR.

#### Testing scripts

```Shell
# Super-resolution scaling factor x2, x3, x4

# scale x2
python main.py --model DDHRN --scale 2 --reset --n_GPUs 1 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_test Set5+Set14+B100+Urban100 --pre_train ../model/DDHRN_scalex2.pt --test_only --save_results

# scale x3
python main.py --model DDHRN --scale 3 --reset --n_GPUs 1 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_test Set5+Set14+B100+Urban100 --pre_train ../model/DDHRN_scalex3.pt --test_only --save_results

# scale x4
python main.py --model DDHRN --scale 4 --reset --n_GPUs 1 --n_resblocks 10 --n_resgroups 10 --n_feats 64 --data_test Set5+Set14+B100+Urban100 --pre_train ../model/DDHRN_scalex4.pt --test_only --save_results
```
### Results
1. For quantitative results, please refer to our paper for more details.
2. You can download some of the testing results of HARTnet from [here]().

## Reproducing other image restoration tasks
Please checkback again for more image restoration applications, including color image denoising, JPEG image deblocing, and low-light image recovery.

## Acknowledgement

Our code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes.
