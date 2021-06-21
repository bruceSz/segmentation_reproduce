#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 tencent.com, Inc. All Rights Reserved
#
########################################################################

import torch

from torch import nn




#def contraction_block(in, out):
def contraction_block(in_channels, out_channels):

    #
    # conv ->  relu -> bn -> conv -> relu -> bn
    #
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3,3), in_channels=in_channels, out_channels=out_channels),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(kernel_size=(3,3), in_channels=out_channels,out_channels=out_channels),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )

    return block



class extension_block(nn.Module):
    def __init__(self, inc, midc, outc):
        super(extension_block, self).__init__().__init__()

        self.up = nn.ConvTranspose2d(inc, inc//2, kernel_size=(3,3), stride=2, padding=1,
            output_padding=1, dilation=1)

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3,3),in_channels=inc, out_channels=midc),
            nn.ReLU(),
            nn.BatchNorm2d(midc),
            nn.Conv2d(kernel_size=(3,3), in_channels=midc, out_channels=outc),
            nn.ReLU(),
            nn.BatchNorm2d(outc)
        )


    def forward(self, e,d):
        d = self.up(d)
        diffY = e.size()[2] - d.size()[2]
        diffX = e.size()[3] - e.size()[3]
        e = e[:,:,diffY//2:e.size()[2]-diffY//2,diffX//2:e.size()[3]-diffX//2]
        cat = torch.cat([e,d],dim=1)
        out = self.block(cat)
        return out


def final_block(inc, outc):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(1,1), in_channels=inc, out_channels=outc),
        nn.ReLU(),
        nn.BatchNorm2d(outc)
    )
    return block


class UNet(nn.Module):
    def __init__(self, inc,outc):
        super(UNet, self).__init__().__init__()
        self.debug = True
        
        #1. encoding
        self.conv_end1 = contraction_block(in_channels=inc, out_channels=  64)
        self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.conv_end2 = contraction_block(in_channels=64, out_channels=  128)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.conv_end3 = contraction_block(in_channels=128,out_channels= 256)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.conv_end4 = contraction_block(in_channels=256, out_channels=512)
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride = 2)


        # bottleneck

        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )
    

        #2. decoding
        self.conv_dec4 = extension_block(1024, 512,512)
        self.conv_dec3 = extension_block(512,256,256)
        self.conv_dec2 = extension_block(256,128,128)
        self.conv_dec1 = extension_block(128,64,64)
        self.final = final_block(64, outc)



    def forward(self, x):

        # encode phase.
        encod1 = self.conv_end1(x) 
        if self.debug: print("encod1 size; ",encod1.size())
        encod_pool1 = self.conv_pool1(encod1)
        

        encod2 = self.conv_end2(encod_pool1)
        encod_pool2 = self.conv_pool2(encod2)
        if self.debug: print("encod2 size; ",encod2.size())

        encod3 = self.conv_end3(encod_pool2)
        encod_pool3 = self.conv_pool3(encod3)
        if self.debug: print("encod3 size; ",encod3.size())

        encod4 = self.conv_end4(encod_pool3)
        encod_pool4 = self.conv_pool4(encod4)
        if self.debug: print("encod4 size; ",encod4.size())


        # bottleneck phase
        bn = self.bottleneck(encod_pool4)

        # decode phase
        dec4 = self.conv_dec4(encod4, bn)
        dec3 = self.conv_dec3(encod3, dec4)
        dec2 = self.conv_dec2(encod2, dec3)
        dec1 = self.conv_dec1(encod1,dec2)

        fl = self.final(dec1)

        return fl


if __name__ == "__main__":
    import torch as t
    rgb = t.randn(1,3,572,572)
    net = UNet(3,12)
    out = net(rgb)
    print(out.shape)