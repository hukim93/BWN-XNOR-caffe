#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_log_dir=examples/cifar10_binary/log $TOOLS/caffe train --gpu=$1 \
    --solver=examples/cifar10_binary/cifar10_full.finetune.solver \
    --weights=examples/cifar10/cifar10_full_iter_40000.caffemodel
