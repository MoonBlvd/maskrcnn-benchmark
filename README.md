# Faster R-CNN and Mask R-CNN in PyTorch 1.0

This repo is forked from https://github.com/facebookresearch/maskrcnn-benchmark. See the [original README](README_old.md) file for details.
The purpose of this repo is to add/modify the original code to train and test on Cityscapes and BDD100K datasets.

## Metrics
The evaluation metrics are following original MaskRCNN paper and the original repo.
1. Average Precision ($AP$)
2. $AP_{50}$: the AP with IOU = 0.5
3. $AP_{75}$: the AP with IOU = 0.75
4. $AP_{S}$: the AP in small scale
5. $AP_{M}$: the AP in median scale
6. $AP_{L}$: the AP in large scale

## Test COCO pretrained model on Cityscapes Dataset
|   Backbone   | EvalType | $AP$    | $AP_{50}$ | $AP_{75}$ | $AP_S$  | $AP_M$  | $AP_L$  |
|:------------:|----------|-------|-------|-------|-------|-------|-------|
| ResNet50-FCN | Bbox     | 0.205 | 0.344 | 0.211 | 0.083 | 0.225 | 0.351 |
| ResNet50-FCN | Mask     | 0.170 | 0.306 | 0.163 | 0.044 | 0.167 | 0.326 |

## Train on CityScapes Dataset

## Test on BDD100K dataset