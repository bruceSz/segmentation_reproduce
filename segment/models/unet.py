#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 tencent.com, Inc. All Rights Reserved
#
########################################################################

import torch

from torch import nn




def contraction_block(in ,out):

    #
    # conv ->  relu -> bn -> conv -> relu -> bn
    #
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3,3), in_channels=in, out_channels=out),
        nn.ReLU(),
        nn.BatchNorm2d(out),
        nn.Conv2d(kernel_size=(3,3), in_channels=out,out_channels=out),
        nn.ReLU(),
        nn.BatchNorm2d(out)
    )

    return block



