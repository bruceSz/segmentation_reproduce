#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2021 tencent.com, Inc. All Rights Reserved
#
########################################################################

import torch
from torch.nn.functional import pad
from torch.nn.modules.conv import ConvTranspose2d
import torchvision.models as models
from torch import nn
from torchvision.models import vgg


vgg16_pretrained = models.vgg16(pretrained=True)



def decoder(inc, outc, num=3):
    if num==3:
        decoder_b = nn.Sequential(
            nn.ConvTranspose2d(inc, inc, 3, padding=1),
            nn.ConvTranspose2d(inc, inc, 3, padding=1),
            nn.ConvTranspose2d(inc, outc, 3, padding=1))
    elif num == 2:
        decoder_b = nn.Sequential(
            nn.ConvTranspose2d(inc, inc, 3, padding=1),
            nn.ConvTranspose2d(inc, outc, 3, padding=1))
    
    else:
        raise RuntimeError("unsupported num")
    return decoder_b


class VGG16_deconv(nn.Module):
    def __init__(self):
        super(VGG16_deconv, self).__init__()
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


        self.clss  = nn.Sequential(
            nn.Linear(512*11*15, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512*11*15),
            nn.ReLU(),
        )


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


    def forward(self,x):
        # 3 * 352 * 480
        en1 = self.encod1(x) 
        o_size1 = en1.size()
        # 64 * 352 * 480
        pl1, indices1 = self.pool1(en1)


        # 64 * 176 * 240
        en2 = self.encod2(x) 
        o_size2 = en2.size()
        # 128 * 176 * 240
        pl2, indices2 = self.pool2(en2)


        # 64 * 176 * 240
        en3 = self.encod3(x) 
        o_size3 = en3.size()
        # 128 * 176 * 240
        pl3, indices3 = self.pool3(en3)


        # 64 * 176 * 240
        en4 = self.encod4(x) 
        o_size4 = en4.size()
        # 128 * 176 * 240
        pl4, indices4 = self.pool4(en4)


        # 64 * 176 * 240
        en5 = self.encod5(x) 
        o_size5 = en5.size()
        # 128 * 176 * 240
        pl5, indices5 = self.pool5(en5)


        # flatten the pool5
        pool5 = pl5.view(pl5.size(0), -1)
        fc = self.clss(pool5)
        fc = fc.reshape(1,512,11,15)



        unpool5 = self.unpool5(input=fc, indices= indices5, output_size=o_size5)
        decoder5 = self.dec5(unpool5)

        unpool4 = self.unpool4(input=decoder5, indices=indices4,output_size=o_size4)
        decoder4 = self.dec4(unpool4)

        unpool3 = self.unpool3(input=decoder4, indices=indices3,output_size=o_size3)
        decoder3 = self.dec3(unpool3)

        unpool2 = self.unpool2(input=decoder3, indices=indices2,output_size=o_size2)
        decoder2 = self.dec2(unpool2)

        unpool1 = self.unpool1(input=decoder2, indices=indices1,output_size=o_size1)
        decoder1 = self.dec1(unpool1)

        return decoder1


if __name__ == "__main__":
    #print(vgg16_pretrained)
    
    rgb = torch.randn(1, 3, 352, 480)

    net = VGG16_deconv()

    out = net(rgb)

    print(out.shape)

















