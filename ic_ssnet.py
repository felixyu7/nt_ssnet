import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME
from resnet_block import ResNetBlock

class SparseIceCubeNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, expand=False, D=4):
        nn.Module.__init__(self)
        self.D = D
        
        # network init
        self.network_initialization(in_channels, out_channels, expand, D)
        self.weight_initialization()
        
    def network_initialization(self, in_channels, out_channels, expand, D):
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels, 16, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=8, stride=8, dimension=4),
            ME.MinkowskiDropout(0.5)
            )

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                16, 16, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(
                16, 32, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(
                32, 32, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv5 = nn.Sequential(
            ME.MinkowskiConvolution(
                32, 32, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv6 = nn.Sequential(
            ME.MinkowskiConvolution(
                32, 64, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )
        
        self.conv7 = nn.Sequential(
            ME.MinkowskiConvolution(
                64, 64, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )
        
        self.conv8 = nn.Sequential(
            ME.MinkowskiConvolution(
                64, 64, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )

        self.conv9 = nn.Sequential(
            ME.MinkowskiConvolution(
                64, 128, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )

        self.conv10 = nn.Sequential(
            ME.MinkowskiConvolution(
                128, 128, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )

        self.conv11 = nn.Sequential(
            ME.MinkowskiConvolution(
                128, 128, kernel_size=3, stride=1, dimension=D, dilation = 1, expand_coordinates=expand,
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )
        
        self.glob_maxpool = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(128, out_channels, bias=True)
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.glob_maxpool(x)
        
        # on cpu, store batch ordering for inference. ME bug?
        inds = torch.sort(x.C[:,0])[1]
        
        x = self.final(x)
        # x = self.tanh(x)
        return x, inds

class SparseIceCubeResNet(torch.nn.Module):

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

        # self.res0 = ResNetBlock(self.first_num_filters, self.first_num_filters, kernel_size=3, expand=expand, stride=2)

        self.pool = ME.MinkowskiMaxPooling(kernel_size=8, stride=8, dimension=4)

        self.resnet = []
        for i, planes in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(planes, planes, expand=expand, dropout=dropout))
            m = nn.Sequential(*m)
            self.resnet.append(m)
            m = []
            if i < self.depth-1:
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
        self.dropout = ME.MinkowskiDropout(0.3)
        # self.tanh = ME.MinkowskiTanh()
        self.final = ME.MinkowskiLinear(planes, out_features, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        # x = self.res0(x)
        x = self.pool(x)
        x = self.resnet(x)
        x = self.glob_pool(x)

        # on cpu, store batch ordering for inference. ME bug?
        inds = torch.sort(x.C[:,0])[1]
        # x = self.dropout(x)
        x = self.final(x)
        # x = self.tanh(x)
        return x, inds