import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.tokenizer.quantizer import QuantizeInterpolatedEMAReset

# main class
class ScaleVQ(nn.Module):
    def __init__(
        self,
        scales,
        nb_code_st,
        nb_code_ed,
        code_dim, 
        args, 
        shared_codebook = False,
        phi_k = 3, 
        phi_depth = 2, 
    ):
        super().__init__()

        num_quantizers = len(scales)
        self.beta = args.commit_beta
        self.scales = scales
        self.code_dim = code_dim

        self.nb_codes = [round(nb_code) for nb_code in np.linspace(nb_code_st, nb_code_ed, num_quantizers)]

        if shared_codebook:
            assert nb_code_st == nb_code_ed
            layer = QuantizeInterpolatedEMAReset(nb_code_st, code_dim, args)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeInterpolatedEMAReset(self.nb_codes[i], code_dim, args) for i in range(num_quantizers)])

        self.phis = nn.ModuleList([Phi(code_dim, phi_k, phi_depth) for _ in range(len(self.scales))])
        
        # only used for progressive training
        self.prog_q = -1
    
    
    @torch.no_grad()
    def get_next_autoregressive_input(self, q, z_hat, z_BCs, scales=None):
        if scales == None:
            scales = self.scales
        T = z_hat.size(-1)
        Q = len(scales)

        if q != Q-1:
            # upsampling
            z_hat_q = self.phis[q](F.interpolate(z_BCs, size=(T), mode='linear'))     # conv after upsample
            z_hat.add_(z_hat_q)
            return z_hat, F.interpolate(z_hat, size=(scales[q+1]), mode='area').permute(0, 2, 1).contiguous()
        else:
            z_hat_q = self.phis[q](F.interpolate(z_BCs, size=(T), mode='linear'))
            z_hat.add_(z_hat_q)
            return z_hat, z_hat.permute(0, 2, 1).contiguous()
    
    
class Phi(nn.Module):
    def __init__(self, embed_dim, ks, depth):
        super().__init__()
        blocks = []
        for _ in range(depth):
            block = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2),
                nn.ReLU()
            )
            blocks.append(block)
        self.phi = nn.Sequential(*blocks)
    
    def forward(self, z_hat_BCT):
        return self.phi(z_hat_BCT)
