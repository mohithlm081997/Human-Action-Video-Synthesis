
import os
import argparse
import time
import random
import sys

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import torchlib
import functools

from torchvision.models import resnet18, ResNet18_Weights
# Insert other imports


import warnings
warnings.filterwarnings("ignore")

def _get_norm_layer_2d(norm):
    # if norm == 'none':
    #     return torchlib.Identity
    # elif norm == 'batch_norm':
    #     return nn.BatchNorm2d
    # elif norm == 'instance_norm':
    #     return functools.partial(nn.InstanceNorm2d, affine=True)
    # elif norm == 'layer_norm':
    #     return lambda num_features: nn.GroupNorm(1, num_features)
    # else:
    #     raise NotImplementedError
    return nn.BatchNorm2d

class ConvDecoder(nn.Module):

    '''
    code from: https://github.com/cc-hpc-itwm/UpConv
    '''
    

    def __init__(self,
                 input_dim=128,
                 output_channels=3,
                 dim=32,
                 n_upsamplings=4,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == nn.Identity),
                Norm(out_dim),
                nn.ReLU()
            )

        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 16)
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0))

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 16)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))

        #layers.append(nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
        layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))
        
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x


class ConvEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1,32,kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.Conv2d(32,32,kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.Conv2d(32,32,kernel_size=(5,5), stride=(1,1), padding=(2,2)),

            nn.Conv2d(32,64,kernel_size=(4,4), stride=(2,2), padding=(1,1),bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),


            nn.Conv2d(64,128,kernel_size=(4,4), stride=(2,2), padding=(1,1),bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            

            nn.Conv2d(128,256,kernel_size=(4,4), stride=(2,2), padding=(1,1),bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,64,kernel_size=(4,4), stride=(2,2), padding=(1,1),bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
        )

    def forward(self, x):
        y = self.net(x)
        return y

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc = ConvEncoder()
        self.dec = ConvDecoder(256,output_channels=16,n_upsamplings=2)

        self.fn = nn.Linear(64+27,256)

    def forward(self,x1,x2):

        y = self.enc(x1)
        y = y.reshape(y.shape[0],-1)
        y = torch.concat([y,x2],1)
        y = F.relu(self.fn(y))
        y = torch.unsqueeze(torch.unsqueeze(y,-1),-1)
        y = self.dec(y)

        return y




def main():

    # gen = ConvDecoder(input_dim=155,output_channels=16)

    # x = torch.randn(10,155,1,1)
    # y = gen(x)
    # print(y.shape)

    # print(gen)


    # disc = ConvEncoder()

    # x = torch.randn(10,1,32,32)
    # y = disc(x)

    # print(y.shape)

    auto_encoder = AutoEncoder()

    x = torch.randn(10,1,16,16)
    x1 = torch.randn(10,27)

    y = auto_encoder(x,x1)

    print(y.shape)




if __name__=="__main__":

    main()
