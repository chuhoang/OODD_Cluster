#!/bin/bash
# sh scripts/ood/oe/cifar10_train_oe.sh

GPU=1
CPU=1
node=73

python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_oe.yml \
    configs/networks/wrn.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_oe.yml \
    configs/preprocessors/base_preprocessor.yml \
    --seed 68791354