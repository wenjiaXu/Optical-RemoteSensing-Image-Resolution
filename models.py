#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:31:23 2017

@author: ldy
"""

import torch
import torch.nn as nn
from math import sqrt

import torch.nn.init as init

import torch.nn.functional as F

num = 64

class DwSample(nn.Module):
    def __init__(self, inp, oup, stride, kernal_size = 3, groups=1, BN = False):
        super(DwSample, self).__init__()
        if BN == True:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
                nn.BatchNorm2d(oup),
                nn.PReLU(),
            )
        else:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size-1)/2), groups=groups),
                nn.PReLU(),
            )

    def forward(self, x):
        residual = x
        out = self.conv_dw(x)
        return torch.add(out, residual)

class BasicBlock(nn.Module):
    def __init__(self, inp, oup, stride, kernal_size=3, groups=1, BN = False):
        super(BasicBlock, self).__init__()
        if BN == True:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
                nn.BatchNorm2d(oup),
                nn.PReLU(),
                nn.Conv2d(oup, inp, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
            )
        else:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
                nn.PReLU(),
                nn.Conv2d(oup, inp, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
            )
    def forward(self, x):
        residual = x
        return torch.add(self.conv_dw(x), residual)

class UpSample(nn.Module):
    def __init__(self, f, upscale_factor):
        super(UpSample, self).__init__()

        self.relu = nn.PReLU()
        self.conv = nn.Conv2d(f, f * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pixel_shuffle(x)
        return x

class DMCN_prelu(nn.Module):
    def __init__(self, BN=True, width = 64):
        super(DMCN_prelu, self).__init__()
        self.input1 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.input2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(width)
        self.input3 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(width)
        self.input4 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(width)
        self.input5 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN4 = nn.BatchNorm2d(width)
        self.down_sample1 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv_DW_layers1 = self.make_layer(DwSample, 5, BN, width)

        self.down_sample2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv_DW_layers2 = self.make_layer(DwSample, 2, BN, width)

        self.up_sample1 = UpSample(width,2)

        self.choose1 = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False)
        self.resudial_layers1 = self.make_layer(BasicBlock, 2, BN, width)

        self.up_sample2 = UpSample(width,2)

        self.choose2 = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False)
        self.resudial_layers2 = self.make_layer(BasicBlock, 5, BN, width)

        self.output = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.PReLU()

    def make_layer(self, block, num_of_layer, BN, width):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(width, width, 1, 3, 1, BN))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        s1 = self.relu(self.input1(x))
        s1 = self.input2(s1)
        s1 = self.relu(self.BN1(s1))
        s1 = self.input3(s1)
        s1 = self.relu(self.BN2(s1))
        s1 = self.input4(s1)
        s1 = self.relu(self.BN3(s1))
        s1 = self.input5(s1)
        s1 = self.relu(self.BN4(s1))
        out = self.down_sample1(s1)
        s2 = self.Conv_DW_layers1(out)

        out = self.down_sample2(s2)
        out = self.Conv_DW_layers2(out)

        out = self.up_sample1(out)
        out = torch.cat((s2, out), 1)
        out = self.choose1(out)
        out = self.resudial_layers1(out)

        out = self.up_sample2(out)
        out = torch.cat((s1, out), 1)
        out = self.choose2(out)
        out = self.resudial_layers2(out)

        out = self.output(out)
        out = torch.add(out, residual)
        return out
