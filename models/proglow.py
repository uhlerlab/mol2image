import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .proglow_modules import CondPrior, ProFlow, SqueezeAvgDiff
from .glow import gaussian_log_p, gaussian_sample
from .condglow import CondGlowMPN
from .mpn import get_MPN

class CondProBlock(nn.Module):
    ''' Modified glow block module for progressive training'''
    def __init__(self, in_channel, n_flow, affine=True, conv_lu=True, n_cond=512):
        super().__init__()

        self.squeeze = SqueezeAvgDiff()

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(ProFlow(in_channel, affine=affine, conv_lu=conv_lu))
        
        self.prior = CondPrior(in_channel + n_cond, in_channel * 6)

    def forward(self, input, cond):
        out, logdet = self.squeeze(input)
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        out, z_new = out

        cond = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, out.size(2), out.size(3))
        mean, log_sd = self.prior(torch.cat((out, cond), 1))
        log_p = gaussian_log_p(z_new, mean, log_sd)
        
        if not np.all(np.isfinite(log_p.data.cpu().numpy())):
            print("Encountered non-finite value in gaussian log p")

        log_p = log_p.view(z_new.size(0), -1).sum(1)

        return out, logdet, log_p, z_new

    def reverse(self, output, cond, eps=None, reconstruct=False):
        if reconstruct:
            z_new = eps

        else:
            cond = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output.size(2), output.size(3))
            mean, log_sd = self.prior(torch.cat((output, cond), 1))
            z_new = gaussian_sample(eps, mean, log_sd)

        # reverse flows
        out = (output, z_new)
        for flow in self.flows[::-1]:
            out = flow.reverse(out)

        # reverse squeeze
        out = self.squeeze.reverse(out)

        return out

class CondProGlow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, pt_config=None, pretrained=None, n_cond=512):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.n_cond = n_cond
        self.pretrained = pretrained
        n_channel = in_channel
        
        for i in range(n_block):
            self.blocks.append(CondProBlock(n_channel, n_flow, affine=affine, conv_lu=conv_lu, n_cond=n_cond))

        if pretrained is not None:
            self.blocks.load_state_dict(torch.load(pretrained)['state_dict'])
            print("Loaded checkpoint from %s" % pretrained)

        if pt_config is not None:
            pt_model = build_proglow_model(pt_config['modules'])
        else:
            pt_model = None

        self.base_blocks = pt_model

    def forward(self, input, cond, return_out=False):

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

        if return_out:
            return log_p_sum, logdet, z_outs, out

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, cond, reconstruct=False):

        z_list_1, z_list_2 = z_list[:len(self.blocks)], z_list[len(self.blocks):]
        input = self.base_blocks.reverse(z_list_2, cond=cond, reconstruct=reconstruct)
        input = torch.clamp(input, min=-0.5, max=0.5)

        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(input, cond=cond, eps=z_list_1[-(i + 1)], reconstruct=reconstruct)

        return input

    def freeze_base_blocks(self):
        for p in self.base_blocks.parameters():
            p.requires_grad = False

class CondProGlowMPN(CondProGlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cond_embedding = nn.Sequential(get_MPN(depth=5, hidden_size=self.n_cond),
                                            nn.Linear(self.n_cond, self.n_cond),
                                            )
        if self.pretrained is not None:
            self.cond_embedding.load_state_dict(torch.load(self.pretrained)['mol_embedding_net'])
            print("Loaded mol checkpoint from %s" % self.pretrained)
        
        self.cond_reduce = nn.Linear(self.n_cond, 15)
        
        if self.pretrained is not None:
            self.cond_reduce.load_state_dict(torch.load(self.pretrained)['mol_reduce_net'])

    def get_mpn_projection(self, cond):
        cond = self.cond_embedding(cond)
        cond = self.cond_reduce(cond)
        norm = cond.pow(2).sum(1, keepdim=True).pow(1. / 2)
        return cond.div(norm)

    def forward(self, input, cond):
        cond = self.cond_embedding(cond)
        return super().forward(input, cond)

    def reverse(self, z_list, cond, reconstruct=False):

        z_list_1, z_list_2 = z_list[:len(self.blocks)], z_list[len(self.blocks):]
        input = self.base_blocks.reverse(z_list_2, cond=cond, reconstruct=reconstruct)
        input = torch.clamp(input, min=-0.5, max=0.5)

        cond = self.cond_embedding(cond)
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(input, cond=cond, eps=z_list_1[-(i + 1)], reconstruct=reconstruct)

        return input


def build_proglow_model(modules):

    if len(modules) == 1:
        m = modules[0]
        assert(m['type'] == 'glow' or '_glow' in m['type'])
    elif len(modules) > 1:
        m = modules[0]
        modules = modules[1:]
    else:
        raise Exception("Length of modules must be at least 1, got length = %s" % len(modules))
    
    if m['type'] == 'cond_glowmpn':
        net = CondGlowMPN(in_channel=m['in_channel'], n_flow=m['n_flow'], n_block=m['n_block'], affine=True, conv_lu=True, n_cond=m['n_cond'])
        if m['checkpoint'] is not None:
            net.load_state_dict(torch.load(m['checkpoint'])['state_dict'])
            print("Loaded checkpoint from %s" % m['checkpoint'])
            
    elif m['type'] == 'cond_proglowmpn':
        net = CondProGlowMPN(in_channel=m['in_channel'], n_flow=m['n_flow'], n_block=m['n_block'], affine=True, conv_lu=True, pretrained=m['checkpoint'], pt_config={'modules': modules}, n_cond=m['n_cond'])
    
    else:
        raise Exception("Model type %s not understood" % m['type'])

    return net

