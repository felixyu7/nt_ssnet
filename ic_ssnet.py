import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import MinkowskiEngine as ME

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
                in_channels, 16, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=8, stride=8, dimension=4),
            ME.MinkowskiDropout(0.5)
            )

        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                16, 16, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(
                16, 32, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv4 = nn.Sequential(
            ME.MinkowskiConvolution(
                32, 32, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv5 = nn.Sequential(
            ME.MinkowskiConvolution(
                32, 32, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.3)
            )
        
        self.conv6 = nn.Sequential(
            ME.MinkowskiConvolution(
                32, 64, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )
        
        self.conv7 = nn.Sequential(
            ME.MinkowskiConvolution(
                64, 64, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=4),
            ME.MinkowskiDropout(0.1)
            )
        
        self.conv8 = nn.Sequential(
            ME.MinkowskiConvolution(
                64, 64, kernel_size=3, stride=1, dimension=D, expand_coordinates=expand, dilation = 1
            ),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiDropout(0.1)
            )
        
        self.glob_maxpool = ME.MinkowskiGlobalMaxPooling()
        self.final = ME.MinkowskiLinear(64, out_channels, bias=True)
        
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
        x = self.glob_maxpool(x)
        
        # on cpu, store batch ordering for inference. ME bug?
        inds = torch.sort(x.C[:,0])[1]
        
        x = self.final(x)
        return x, inds
        
        