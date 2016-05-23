#!/usr/bin/env sh

if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

TOOLS=./build/tools

$TOOLS/caffe train \
    --gpu=$gpu \
    --solver=examples/cifar10_acc/cifar10_conv2.finetune.solver \
    --weights=examples/cifar10/cifar10_full_iter_40000.caffemodel
