#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 tencent.com, Inc. All Rights Reserved
#
########################################################################

from torch import nn

class Block(nn.Module):
    def __init__(self, inc, outc, reps, stride=1, dilation=1,start_with_relu=True,grow_first=True, is_last=False):
        super(Block, self).__init__()
        if inc != outc or stride!=1:
            self.skip = nn.Conv2d(inc, outc,1, stride=stride,bias=False)
            self.skipbn = nn.BatchNorm2d(outc)
        else:
            self.skip = None
        self.relu = nn.Relu(inplace=True)
        rep = []

        filters = inc
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(inc,outc,3,stride=1,dilation=dialtion))
            rep.append(nn.BatchNorm2d(outc))
            filters = outc



class  Xception(nn.Module):
    def __init__(self, inc=3, os=16):
        super(Xception, self).__init__()
        if os==16:
            entry_block3_stride=2
            middle_block_dilation= 1
            exit_block_dilation=(1,2)
        elif os== 8:

            entry_block3_stride=1
            middle_block_dilation= 2
            exit_block_dilation=(2,4)
        else:
            raise NotImplementedError
        
        self.conv1 = nn.Conv2d(inc, 32,3,stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64,128,reps=2,sride=2,start_with_relu=False)


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus,self).__init__(self, inc=3, nclass=12, os=16, _print=True)
        if _print:
            pass
        self.xception = Xception(inc, os)


if __name__ == "__main__":
    model = DeepLabV3Plus(inc =3, nclass=12, os=16, _print=True)
    model.eval()
    image = torch.randn(1,3,352,480)
    out = model(image)
    print(out.size())
