""" ic_ssnet.py - Network class for SSCNN
    Felix J. Yu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
from resnet_block import ResNetBlock

class SparseIceCubeResNet(torch.nn.Module):

    """ SSCNN class. Utilizes a ResNet structure with blocks. Implemented using MinkowskiEngine. """

    def __init__(self, in_features, out_features, reps=2, depth=8, first_num_filters=16, stride=2, expand=False, dropout=0., D=4):
        super(SparseIceCubeResNet, self).__init__()

        self.D = D
        self.reps = reps
        self.depth = depth
        self.first_num_filters = first_num_filters
        self.stride = stride
        self.nPlanes = [i * self.first_num_filters for i in range(1, self.depth + 1)]

        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_features,
            out_channels=self.first_num_filters,
            kernel_size=3, stride=3, dimension=self.D, dilation=1,
            bias=False)

        self.pool = ME.MinkowskiMaxPooling(kernel_size=8, stride=8, dimension=4)

        self.resnet = []
        for i, planes in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(planes, planes, expand=expand, dropout=dropout))
            m = nn.Sequential(*m)
            self.resnet.append(m)
            m = []
            if i < self.depth - 1:
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D, expand_coordinates=expand,
                    bias=False))
                m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
                m.append(ME.MinkowskiPReLU())
            m = nn.Sequential(*m)
            self.resnet.append(m)
        self.resnet = nn.Sequential(*self.resnet)
        self.glob_pool = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(planes, out_features, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool(x)
        x = self.resnet(x)
        x = self.glob_pool(x)

        # on cpu, store batch ordering for inference. ME bug?
        inds = torch.sort(x.C[:,0])[1]
        x = self.final(x)

        return x, inds