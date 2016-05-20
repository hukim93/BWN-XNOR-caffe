#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_log_dir=examples/cifar10_xnor/log $TOOLS/caffe train --gpu $1 \
    --solver=examples/cifar10_xnor/cifar10_full.solver
