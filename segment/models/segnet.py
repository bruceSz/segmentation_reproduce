#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 tencent.com, Inc. All Rights Reserved
#
########################################################################



import torch
import torchvision.models  as models

from torch import nn 

def decoder(inc, outc, num=3):
    if num==3:
        db = nn.Sequential(
            nn.Conv2d(inc, inc, 3, padding=1),
            nn.Conv2d(inc, inc, 3, padding=1),
            nn.Conv2d(inc, outc, 3, padding=1))
    elif num == 2:
        db = nn.Sequential(
            nn.Conv2d(inc, inc, 3, padding=1),
            nn.Conv2d(inc, outc, 3, padding=1))
    else:
        raise RuntimeError("invalid num")
    return db



class VGG16_segnet(nn.Module):
    VGG_16 = models.vgg16(pretrained=True)    
    def __init__(self):
        super(VGG16_segnet, self).__init__()
        vgg16_pretrained = self.VGG_16
        
        in_l = [4,9,16,23,30]
        for index in in_l:
            vgg16_pretrained.features[index].return_indices = True
        
        self.encod1 = vgg16_pretrained.features[:4]
        self.pool1 = vgg16_pretrained.features[4]

        self.encod2 = vgg16_pretrained.features[5:9]
        self.pool2 = vgg16_pretrained.features[9]

        self.encod3 = vgg16_pretrained.features[10:16]
        self.pool3 = vgg16_pretrained.features[16]

        self.encod4 = vgg16_pretrained.features[17:23]
        self.pool4 = vgg16_pretrained.features[23]

        self.encod5 = vgg16_pretrained.features[24:30]
        self.pool5 = vgg16_pretrained.features[30]

        


        self.dec5 = decoder(512,512)
        self.unpool5 = nn.MaxUnpool2d(2,2)

        self.dec4 = decoder(512,256)
        self.unpool4 = nn.MaxUnpool2d(2,2)

        self.dec3 = decoder(256,128)
        self.unpool3 = nn.MaxUnpool2d(2,2)


        self.dec2 = decoder(128,64,2)
        self.unpool2 = nn.MaxUnpool2d(2,2)

        self.dec1 = decoder(64,12,2)
        self.unpool1 = nn.MaxUnpool2d(2,2)

    def forward(self, x):
        # 3 * 352 * 480
        enc1 = self.encod1(x)
        o_size1 = enc1.size()
        # 64 * 352 * 480
        pl1, ind1 = self.pool1(enc1)


        # 64 * 176 * 240
        enc2 = self.encod2(pl1)
        o_size2 = enc2.size()
        # 128 * 176 * 240
        pl2, ind2 = self.pool2(enc2)


        # 128 * 88 * 120
        enc3 = self.encod3(pl2)
        o_size3 = enc3.size()
        # 256 * 88 * 120
        pl3, ind3 = self.pool3(enc3)


        # 256 * 44 * 60
        enc4 = self.encod4(pl3)
        o_size4 = enc4.size()
        # 512 * 44 * 60
        pl4, ind4 = self.pool4(enc4)

        # 512 * 22 * 30
        enc5 = self.encod5(pl4)
        o_size5 = enc5.size()
        # 512 * 11 * 15
        pl5, ind5 = self.pool5(enc5)

        # decoder

        up5 = self.unpool5(input=pl5,indices=ind5, output_size=o_size5)
        dec5 = self.dec5(up5)

        up4 = self.unpool4(input=dec5,indices=ind4, output_size=o_size4)
        dec4 = self.dec4(up4)


        up3 = self.unpool3(input=dec4,indices=ind3, output_size=o_size3)
        dec3 = self.dec3(up3)

        up2 = self.unpool2(input=dec3,indices=ind2, output_size=o_size2)
        dec2 = self.dec2(up2)

        up1 = self.unpool1(input=dec2,indices=ind1, output_size=o_size1)
        dec1 = self.dec1(up1)

        return dec1

if __name__ == "__main__":
    import torch as t

    rgb = t.randn(1, 3, 352, 480)

    net = VGG16_segnet()

    out = net(rgb)

    print(out.shape)