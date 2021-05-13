# encoding: utf-8

import numpy as np
import torch
from torchvision import models

from torch import nn



def bilinear_kernel(in_c, out_c, k_size):
    factor = (k_size + 1)//2
    


class FCN(nn.Module):
    def __init__(self, num_c):
        super().__init__()
        self.vgg = models.vgg16_bn(pretrained=False)

        self.stg1 = self.vgg.features[0:7]
        self.stg2 = self.vgg.features[7:14]
        self.stg3 = self.vgg.features[14:24]
        self.stg4 = self.vgg.features[24:34]
        self.stg5 = self.vgg.features[34:]


        self.score1 = nn.Conv2d(512,num_c, 1)
        self.score2 = nn.Conv2d(512, num_c, 1)
        self.score3 = nn.Conv2d(512, num_c, 1)

        self.conv_trans1 = nn.Conv2d(512,256,1)
        self.conv_trans2 = nn.Conv2d（256，num_c, 1)

        self.upsample_8x = nn.ConvTranspose2d(num_c,num_c,16,8,4,bias=True)
        self.upsample_8x.weight.data = bilinear_kernel(num_c,num_c,16)

        self.upsample_2x_1 = nn.ConvTranspose2d(512, 512,4,2,1,bias=True)
        self.upsample_2x_1.weight.data = bilinear_kernel(512,512,4)

        self.upsample_2x_2 = nn.ConvTranspose2d(256,256,4,2,1,bias=True)
        self.upsample_2x_2.weight.data = bilinear_kernel(256,256,4)






