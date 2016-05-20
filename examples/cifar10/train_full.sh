#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_log_dir=examples/cifar10/log $TOOLS/caffe train --gpu $1 \
    --solver=examples/cifar10/cifar10_full.solver
