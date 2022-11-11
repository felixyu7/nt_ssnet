import torch.nn as nn
import torch
import MinkowskiEngine as ME

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input

class ResNetBlock(ME.MinkowskiNetwork):
    '''
    ResNet Block with Leaky ReLU nonlinearities.
    '''
    expansion = 1

    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 dilation=1,
                 kernel_size=3,
                 dimension=4,
                 dropout=0.,
                 bias=False):
        super(ResNetBlock, self).__init__(dimension)
        assert dimension > 0

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features, 
                                               bias=bias)
        else:
            self.residual = Identity()
        
        self.conv1 = nn.Sequential(ME.MinkowskiConvolution(
            in_features, out_features, kernel_size=kernel_size,
            stride=stride, dilation=dilation, dimension=dimension, bias=bias),
            ME.MinkowskiBatchNorm(out_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiDropout(dropout))

        self.conv2 = nn.Sequential(ME.MinkowskiConvolution(
            out_features, out_features, kernel_size=kernel_size,
            stride=stride, dilation=dilation, dimension=dimension, bias=bias),
            ME.MinkowskiBatchNorm(out_features),
            ME.MinkowskiReLU(),
            ME.MinkowskiDropout(dropout))
        
        self.weight_initialization()
        
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual
