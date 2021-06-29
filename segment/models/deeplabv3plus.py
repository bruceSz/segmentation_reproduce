#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 tencent.com, Inc. All Rights Reserved
#
########################################################################

from torch import nn
from torch._C import strided


class Block(nn.Module):
    def __init__(self, inc, outc, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if inc != outc or stride != 1:
            self.skip = nn.Conv2d(inc, outc, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(outc)
        else:
            self.skip = None
        self.relu = nn.Relu(inplace=True)
        rep = []

        filters = inc
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(
                inc, outc, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(outc))
            filters = outc
        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(
                filters, filters, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(outc))
            filters = outc

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d_same(
                inc, outc, 3, stride=1, dilation=dilation))
            rep.append(nn.BatchNorm2d(outc))

        if not start_with_relu:
            rep = ret[1:]

        if stride != 1:
            rep.append(SeparableConv2d_same(outc, outc, 3, stride=2))
        if stride == 1 and is_last:
            rep.append(SeparableConv2d_same(outc, outc, 3, stride=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    def __init__(self, inc=3, os=16):
        super(Xception, self).__init__()
        if os == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilation = (1, 2)
        elif os == 8:

            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilation = (2, 4)
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(inc, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, reps=2, sride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2,
                            start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, stride=entry_block3_stride,
                            start_with_relu=True, grow_first=True, is_last=True)

        self.block4 = Block(728, 728, reps=3, stride=1,
                            dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, reps=3, stride=1,
                            dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, reps=3, stride=1,
                            dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, reps=3, stride=1,
                            dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, reps=3, stride=1,
                            dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, reps=3, stride=1,
                            dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1,
                             dilation=middle_block_dilation, start_with_relu=True, grow_first=True)

        self.block20 = Block(728, 1024, reps=2, stride=1,
                             dilation=exit_block_dilations[0], start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d_same(
            1024, 1536, 3, stride=1, dilation=exit_block_dilation[1])
        self.bn3 = nn.BatchNorm2d(1536)

        self.conv4 = SeprableConv2d_same(
            1536, 1536, 3, stride=1, dilation=exit_block_dilation[1])
        self.bn4 = nn.BatchNorm2d(1536)

        self.conv5 = SeparableConv2d_same(
            1536, 2048, stride=1, dilation=exit_block_dilation[1])
        self.bn5 = nn.BatchNorm2d(2048)

        self._init_weight()

    def forward(self, x):
        # entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x

        x = self.block2(x)
        x = self.block3(x)

        # middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self):
        super().__init__(inc, outc, os):
        super(ASPP, self).__init__()

        # ASPP
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 32]

        self.aspp1 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0, dilation=dilations[0], bias=False),
                                   nn.BatchNorm2d(outc),
                                   nn.ReLU())
        self.aspp2 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),
                                   nn.BatchNorm2d(outc),
                                   nn.ReLU())

        self.aspp3 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2], bias=False),
                                   nn.BatchNorm2d(outc),
                                   nn.ReLU())

        self.aspp4 = nn.Sequential(nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[4], bias=False),
                                   nn.BatchNorm2d(outc),
                                   nn.ReLU())

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1,
                                                       stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[
                           2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus, self).__init__(
            self, inc=3, nclass=12, os=16, _print=True)
        if _print:
            pass
        self.xception = Xception(inc, os)


if __name__ == "__main__":
    model = DeepLabV3Plus(inc=3, nclass=12, os=16, _print=True)
    model.eval()
    image = torch.randn(1, 3, 352, 480)
    out = model(image)
    print(out.size())
