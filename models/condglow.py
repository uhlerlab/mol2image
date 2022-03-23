import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .glow import ActNorm, InvConv2d, InvConv2dLU, AffineCoupling, Flow, Prior, gaussian_log_p, gaussian_sample, CondFlow
from .mpn import get_MPN

class CondBlock(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True, n_cond=512):
        super().__init__()
        
        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = Prior(in_channel * 2 + n_cond, in_channel * 4)

        else:
            self.prior = Prior(in_channel * 4 + n_cond, in_channel * 8)

    def forward(self, input, cond):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det
            
        cond = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height // 2, width // 2)

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(torch.cat((out, cond), 1)).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(torch.cat((zero, cond), 1)).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, cond, eps=None, reconstruct=False):
        ''' Reverses block of flow 
        Parameters:

        output (torch.FloatTensor): latent dims passed up from previous block
        eps (torch.FloatTensor): latent dims to concatenate
        reconstruct (bool): whether to reconstruct image from latents (True) or sample new image (False)

        Returns:
        torch.FloatTensor: output from reversing flow transformation
        '''

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            cond = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output.size(2), output.size(3))

            if self.split:
                mean, log_sd = self.prior(torch.cat((output, cond), 1)).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(output)
                mean, log_sd = self.prior(torch.cat((zero, cond), 1)).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed

class CondGlow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, n_cond=512):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.n_cond = n_cond

        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(CondBlock(n_channel, n_flow, affine=affine, conv_lu=conv_lu, n_cond=n_cond))
            n_channel *= 2
        self.blocks.append(CondBlock(n_channel, n_flow, split=False, affine=affine, n_cond=n_cond))

    def forward(self, input, cond):

        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out, cond)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, cond, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], cond=cond, eps=z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, cond=cond, eps=z_list[-(i + 1)], reconstruct=reconstruct)

        return input

class CondGlowMPN(CondGlow):
    '''Extension of CondGlow that embeds the conditioning feature'''
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, n_cond=512):
        super().__init__(in_channel, n_flow, n_block, affine=affine, conv_lu=conv_lu, n_cond=n_cond)

        self.cond_embedding = nn.Sequential(get_MPN(depth=5, hidden_size=self.n_cond),
                                            nn.Linear(self.n_cond, self.n_cond),
                                            )
        self.cond_reduce = nn.Linear(self.n_cond, calc_reduce_size(n_channel=in_channel, n_block=n_block))

    def get_mpn_projection(self, cond):
        cond = self.cond_embedding(cond)
        cond = self.cond_reduce(cond)
        norm = cond.pow(2).sum(1, keepdim=True).pow(1. / 2)
        return cond.div(norm)

    def forward(self, input, cond):
        cond = self.cond_embedding(cond)
        return super().forward(input, cond)

    def reverse(self, z_list, cond, reconstruct=False):
        cond = self.cond_embedding(cond)
        return super().reverse(z_list, cond, reconstruct=reconstruct)

def calc_reduce_size(n_channel, n_block):
    reduce_size = 0
    for i in range(n_block - 1):
        n_channel *= 2
        reduce_size += n_channel
    reduce_size += n_channel * 4
    return reduce_size
