import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .glow import ActNorm, InvConv2dLU, InvConv2d, AffineCoupling, gaussian_sample, gaussian_log_p, CondAffineCoupling

class AffineCoupling2(AffineCoupling):
    def __init__(self, in_channel, filter_size=128, affine=True):
        super().__init__(in_channel=in_channel, filter_size=filter_size, affine=affine)

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size*2, 3, padding=1),
            nn.BatchNorm2d(filter_size*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size*2, filter_size*4, 3, padding=1),
            nn.BatchNorm2d(filter_size*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size*4, in_channel if self.affine else in_channel // 2, 1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0, 0.05)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

class CondAffineCoupling2(CondAffineCoupling):
    def __init__(self, in_channel, filter_size=128, affine=True, n_cond=512):
        super().__init__(in_channel=in_channel, filter_size=filter_size, affine=affine, n_cond=n_cond)

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2 + n_cond, filter_size, 3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size*2, 3, padding=1),
            nn.BatchNorm2d(filter_size*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size*2, filter_size*4, 3, padding=1),
            nn.BatchNorm2d(filter_size*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size*4, in_channel if self.affine else in_channel // 2, 1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0, 0.05)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

class CondPrior(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channel, filter_size, 3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size*2, 3, padding=1),
            nn.BatchNorm2d(filter_size*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size*2, filter_size*4, 3, padding=1),
            nn.BatchNorm2d(filter_size*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size*4, out_channel, 1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                layer.weight.data.normal_(0, 0.05)
                layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1.0)
                nn.init.constant_(layer.bias, 0.0)
        
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, cond):
        out = self.net(cond)
        mu, log_sd = out.chunk(2, 1)

        # activation for stability
        log_sd = 2*torch.tanh(log_sd)
        return mu, log_sd

class ProFlow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.in_channel = in_channel
        self.actnorm = ActNorm(in_channel*3)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel*3)

        else:
            self.invconv = InvConv2d(in_channel*3)

        self.coupling = AffineCoupling2(in_channel*4, affine=affine)

    def forward(self, input):
        input1, residual = input
        residual, logdet = self.actnorm(residual)
        residual, det1 = self.invconv(residual)
        out, det2 = self.coupling(torch.cat((input1, residual), dim=1))
        out = (out[:, 0:self.in_channel, :, :], out[:, self.in_channel:, :, :])

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(torch.cat(output, dim=1))
        input1, residual = input[:, 0:self.in_channel, :, :], input[:, self.in_channel:, :, :]
        residual = self.invconv.reverse(residual)
        residual = self.actnorm.reverse(residual)
        return (input1, residual)

class CondProFlow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True, n_cond=512):
        super().__init__()

        self.in_channel = in_channel
        self.actnorm = ActNorm(in_channel*3)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel*3)

        else:
            self.invconv = InvConv2d(in_channel*3)

        self.coupling = CondAffineCoupling2(in_channel*4, n_cond=n_cond, affine=affine)

    def forward(self, input, cond):
        input1, residual = input
        residual, logdet = self.actnorm(residual)
        residual, det1 = self.invconv(residual)
        out, det2 = self.coupling(torch.cat((input1, residual), dim=1), cond)
        out = (out[:, 0:self.in_channel, :, :], out[:, self.in_channel:, :, :])

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output, cond):
        input = self.coupling.reverse(torch.cat(output, dim=1), cond)
        input1, residual = input[:, 0:self.in_channel, :, :], input[:, self.in_channel:, :, :]
        residual = self.invconv.reverse(residual)
        residual = self.actnorm.reverse(residual)
        return (input1, residual)

class SqueezeAvgDiff(nn.Module):
    ''' Module for spatially squeezing and transforming channels '''
    def __init__(self):
        super().__init__()
        
        weight = [[ 1/4, 1/4, 1/4, 1/4],
                  [ 1/4, 1/4,-1/4,-1/4],
                  [ 1/4,-1/4, 1/4,-1/4],
                  [ 1/4,-1/4,-1/4, 1/4],]

        weight = torch.FloatTensor(weight).view(4, 4, 1, 1, 1)
        self.register_buffer('weight', weight)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 3, 5, 1, 2, 4)
        squeezed = squeezed.contiguous().view(b_size, 4, n_channel, height // 2, width // 2)

        out = F.conv3d(squeezed, self.weight)
        # separate first channel from others
        out = (out[:, 0, :, :, :].squeeze(1), out[:, 1:, :, :, :].view(b_size, 3*n_channel, height // 2, width // 2))
        
        logdet = (
            1/4 * n_channel * height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, input):
        # reshape and stack the two input blocks
        b_size, n_channel, height, width = input[1].shape
        n_channel = n_channel // 3
        input = (input[0].unsqueeze(1), input[1].view(b_size, 3, n_channel, height, width))
        input = torch.cat(input, dim=1)

        # inverse linear transformation
        out = F.conv3d(input, self.weight.squeeze().inverse().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        # reverse spatial squeezing operation
        out = out.view(b_size, 2, 2, n_channel, height, width)
        out = out.permute(0, 3, 4, 1, 5, 2)
        out = out.contiguous().view(b_size, n_channel, height*2, width*2)

        return out

