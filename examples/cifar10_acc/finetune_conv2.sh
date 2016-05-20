#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --gpu=$1 \
    --solver=examples/cifar10_acc/cifar10_conv2.finetune.solver \
    --weights=examples/cifar10/cifar10_full_iter_90000.caffemodel
