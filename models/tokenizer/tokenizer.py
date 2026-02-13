import torch
import torch.nn as nn
import numpy as np

from models.tokenizer.encdec import Encoder, Decoder
from models.tokenizer.vq import ScaleVQ
from utils.tools import lengths_to_mask
    
class SQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width,
                 scales,
                 nb_code_st = 256,
                 nb_code_ed = 768,
                 code_dim = 512,
                 width = 256,
                 width_mul = (1, 2, 4),
                 depth = 2,
                 slot_group = 10,
                 ):

        super().__init__()
        self.input_width = input_width
        self.code_dim = code_dim
        self.mu = args.mu
        self.group_num = slot_group
        self.T = args.max_motion_length
        self.slot_num = args.max_motion_length * slot_group

        self.encoder = Encoder(ch=width, ch_mult=width_mul, num_res_blocks=depth, in_channels=input_width, z_channels=code_dim)
        self.decoder = Decoder(ch=width, ch_mult=width_mul, num_res_blocks=depth, in_channels=input_width, z_channels=code_dim) 

        pqvae_config = {
            'scales': scales,
            'nb_code_st': nb_code_st,
            'nb_code_ed': nb_code_ed,
            'code_dim':code_dim, 
            'args': args,
            'shared_codebook': args.shared_codebook,
            'phi_k': args.phi_k,
            'phi_depth': args.phi_depth,
        }

        self.quantizer = ScaleVQ(**pqvae_config)
        self.reset_slot()
    
    def reset_slot(self):
        self.virtual_slot = nn.Parameter(torch.zeros(self.slot_num, self.input_width), requires_grad=False)
        self.register_buffer('initted', torch.Tensor([False]))
    
    def _tile(self, x, x_lens):
        """
        param: x  (N, T, dim)
        param: x_lens (N, T)
        """
        res_x = x[x_lens>=self.T].reshape(-1, x.shape[-1])
        T_x, code_dim = res_x.shape
        if 0 < T_x < self.slot_num:
            n_repeats = (self.slot_num + T_x - 1) // T_x
            std = 0.01 / np.sqrt(code_dim)
            out = res_x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x.reshape(-1, x.shape[-1])
        return out
        
    def init_slot(self, x, x_lens):
        x = x.transpose(1, 2)
        out = self._tile(x, x_lens)
        self.virtual_slot.data = out[:self.slot_num]
        self.virtual_slot.requires_grad = True
        self.initted = x.new_tensor([True])

    def pad_slot(self, x, x_lens):
        N, dim, T = x.shape
        x = x.transpose(1, 2)

        # concatenate slot
        pad_mask = ~lengths_to_mask(x_lens, max(x_lens))
        idx = torch.randint(0, self.group_num, [1]).item()
        expand_slots = self.virtual_slot.view(self.group_num, -1, dim)[idx:idx+1].expand(N, -1, -1)[:,:max(x_lens)]
        x = x[:, :max(x_lens)]
        x[pad_mask] = expand_slots[pad_mask].to(x.dtype)

        # MoCo update
        if self.training:
            out = self._tile(x, x_lens)
            new_slot = out[:self.slot_num]
            with torch.no_grad():
                self.virtual_slot.data = self.mu * self.virtual_slot.data + (1 - self.mu) * new_slot
            
        return x.transpose(1, 2).contiguous()

    def preprocess(self, x):
        # (bs, T, dim) -> (bs, dim, T)
        x = x.permute(0, 2, 1).float().contiguous()
        return x

    def forward_decoder(self, z_hat):
        m_hat = self.decoder(z_hat)
        return m_hat
