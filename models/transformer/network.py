import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from .embed_rope import apply_rotary_emb

class SwiGLU(nn.Module):
    def __init__(self, latent_dim, ff_size) -> None:
        super().__init__()
        
        self.c_fc1 = nn.Linear(latent_dim, ff_size, bias=False)
        self.c_fc2 = nn.Linear(latent_dim, ff_size, bias=False)
        self.c_proj = nn.Linear(ff_size, latent_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


class SelfAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, freqs_cis, attn_mask=None, padding_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        if freqs_cis is not None:
            q, k= apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=2); v = self.cached_v = torch.cat((self.cached_v, v), dim=2)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # or cross-attention; (B, nh, T, hs) x (B, nh, hs, M) -> (B, nh, T, M)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        if padding_mask is not None:
            att = att.masked_fill(padding_mask[:, None, None, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class CrossAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v = enable, None, None

    def forward(self, x, y, attn_mask=None):
        B, T, C = x.size()
        _, M, _ = y.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.caching:
            if self.cached_k is None: 
                self.cached_k = k = self.key(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
                self.cached_v = v = self.value(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
            else: 
                k = self.cached_k; v = self.cached_v
        else:
            k = self.key(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
            v = self.value(y).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # or cross-attention; (B, nh, T, hs) x (B, nh, hs, M) -> (B, nh, T, M)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask[:, None, None, :] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


# class CrossAttentionPro2(nn.Module):

#     def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
#         super().__init__()
#         assert embed_dim % 8 == 0
#         # key, query, value projections for all heads
#         self.qkv_x = nn.Linear(embed_dim, embed_dim*3)
#         self.qkv_y = nn.Linear(embed_dim, embed_dim*3)

#         self.sattn_drop = nn.Dropout(drop_out_rate)
#         self.cattn_drop = nn.Dropout(drop_out_rate)
#         self.resid_drop = nn.Dropout(drop_out_rate)
#         # self.gate_s = nn.Linear(embed_dim, embed_dim)
#         # self.gate_c = nn.Linear(embed_dim, embed_dim)
     

#         self.proj = nn.Linear(embed_dim, embed_dim)

#         self.n_head = n_head

#         # only used during inference
#         self.caching, self.cached_qkv_y = False, None
    
#     def kv_caching(self, enable: bool): self.caching, self.cached_k_x, self.cached_y_x,  self.cached_qkv_y = enable, None, None, None

#     def forward(self, x, y, attn_x_mask=None):
#         B, T, C = x.size()
#         _, M, _ = y.size()

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         q_x, k_x, v_x = self.qkv_x(x).chunk(3, dim=-1)
#         q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

#         if self.caching and self.cached_qkv_y is not None:
#             q_y, k_y, v_y = self.cached_qkv_y
#         else:
#             q_y, k_y, v_y = self.cached_qkv_y = self.qkv_y(y).chunk(3, dim=-1)

#         if self.caching:
#             if self.cached_k_x is None: self.cached_k_x = k_x; self.cached_v_x = v_x
#             else: k_x = self.cached_k_x = torch.cat((self.cached_k_x, k_x), dim=2); v_x = self.cached_v_x = torch.cat((self.cached_v_x, v_x), dim=2)

#         q_y = q_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
#         k_y = k_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
#         v_y = v_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

#         # cross attention of x: (B, nh, T, hs) x (B, nh, M, hs) -> (B, nh, T, M)
#         catt_x2y = (q_x @ k_y.transpose(-2, -1)) * (1.0 / math.sqrt(k_y.size(-1)))
#         # cross attention of y: (B, nh, M, hs) x (B, nh, T-kv, hs) -> (B, nh, M, T-kv)
#         catt_y2x = (q_y @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
#         # (B, nh, T, M) x (B, nh, M, T-kv) -> (B, nh, T, T-kv)
#         catt_x2x_ = catt_x2x = (catt_x2y @ catt_y2x) * (1.0 / math.sqrt(k_x.size(-1)))
#         if attn_x_mask is not None:
#             catt_x2x_ = catt_x2x.masked_fill(attn_x_mask == 0, float('-inf'))
#         # 训练：(B, nh, T-kv, T-kv) * (B, nh, 1, T-kv) -sum-> (B, nh, T-kv)
#         # 推理：(B, nh, T, T-kv) * (B, nh, 1, T-kv) -sum-> (B, nh, T-kv)
#         catt_x2x_score = torch.sum(F.softmax(catt_x2x_, dim=-1) * catt_x2x, dim=-1, keepdim=True)

#         # (B, nh, M, d) x (B, nh, M, d) -> (B, nh, M, M)
#         catt_y2y = (q_y @ k_y.transpose(-2, -1)) * (1.0 / math.sqrt(k_y.size(-1)))
#         catt_y2y_score = torch.sum(F.softmax(catt_y2y, dim=-1) * catt_y2y, dim=-1, keepdim=True).transpose(-1, -2)

#         # (B, nh, T, T-kv) x (B, nh, T-kv, hs) -> (B, nh, T, hs)
#         cval = self.cattn_drop(F.softmax(catt_x2y, dim=-1) * catt_y2y_score) @ v_y 
#         cval = cval.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

#         # self attention of x: (B, nh, T, hs) x (B, nh, T, hs) -> (B, nh, T, T)
#         satt_x = (q_x @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))

#         if attn_x_mask is not None:
#             satt_x = satt_x.masked_fill(attn_x_mask == 0, float('-inf'))
#         sval = self.sattn_drop(F.softmax(satt_x, dim=-1) * catt_x2x_score) @ v_x
#         sval = sval.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

#         y = sval + cval
#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y
    

class CrossAttentionPro(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.qkv = nn.Linear(embed_dim, embed_dim*3)

        self.cattn_x2y_drop = nn.Dropout(drop_out_rate)
        self.cattn_y2x_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_qkv_y = False, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k_x, self.cached_y_x,  self.cached_qkv_y = enable, None, None, None

    def forward(self, x, y, attn_x_mask=None):
        B, T, C = x.size()
        _, M, _ = y.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q_x, k_x, v_x = self.qkv(x).chunk(3, dim=-1)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.caching and self.cached_qkv_y is not None:
            q_y, k_y, v_y = self.cached_qkv_y
        else:
            q_y, k_y, v_y = self.cached_qkv_y = self.qkv(y).chunk(3, dim=-1)

        if self.caching:
            if self.cached_k_x is None: self.cached_k_x = k_x; self.cached_v_x = v_x
            else: k_x = self.cached_k_x = torch.cat((self.cached_k_x, k_x), dim=2); v_x = self.cached_v_x = torch.cat((self.cached_v_x, v_x), dim=2)

        q_y = q_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        k_y = k_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v_y = v_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        # cross attention of x: (B, nh, T, hs) x (B, nh, M, hs) -> (B, nh, T, M)
        catt_x2y = (q_x @ k_y.transpose(-2, -1)) * (1.0 / math.sqrt(k_y.size(-1)))
        # cross attention score of x2y (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        cval_x2y = self.cattn_x2y_drop(F.softmax(catt_x2y, dim=-1)) @ v_y 
        cval_x2y = cval_x2y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # cross attention of y: (B, nh, M, hs) x (B, nh, T-kv, hs) -> (B, nh, M, T-kv)
        catt_y2x = (q_y @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
        # (B, nh, T, M) x (B, nh, M, T-kv) -> (B, nh, T, T-kv)
        catt_y2x = (catt_x2y @ catt_y2x) * (1.0 / math.sqrt(k_x.size(-1)))
        if attn_x_mask is not None:
            catt_y2x = catt_y2x.masked_fill(attn_x_mask == 0, float('-inf'))
        # (B, nh, T, T-kv) x (B, nh, T-kv, hs) -> (B, nh, T, hs)
        cval_y2x = self.cattn_y2x_drop(F.softmax(catt_y2x, dim=-1)) @ v_x
        cval_y2x = cval_y2x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = cval_x2y - cval_y2x
        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
    

class ConditionGateAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.key_c = nn.Linear(embed_dim, embed_dim)
        self.value_c = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop1 = nn.Dropout(drop_out_rate)
        self.attn_drop2 = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.weight_proj1 = nn.Linear(embed_dim, 1, bias=False)
        self.weight_proj2 = nn.Linear(embed_dim, 1, bias=False)
        self.proj_gate1 = nn.Linear(embed_dim, embed_dim)
        self.proj_gate2 = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v, self.cached_k_c, self.cached_v_c = False, None, None, None,  None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v, self.cached_k_c, self.cached_v_c = enable, None, None, None,  None

    def forward(self, x, c, attn_mask=None, padding_mask=None):
        B, T, C = x.size()
        _, M, _ = c.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        # key-value caching tech.
        if self.caching:
            if self.cached_k is None: 
                self.cached_k, self.cached_v = k, v
                k_c = self.cached_k_c = self.key_c(c).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
                v_c = self.cached_v_c = self.value_c(c).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
            else: 
                k = self.cached_k = torch.cat((self.cached_k, k), dim=2)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=2)
                k_c, v_c = self.cached_k_c, self.cached_v_c
        else:
            k_c = self.key_c(c).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
            v_c = self.value_c(c).view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        # self attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop1(att)
        y = att @ v # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # cross attention
        att_c = (q @ k_c.transpose(-2, -1)) * (1.0 / math.sqrt(k_c.size(-1)))
        if padding_mask is not None:
            att_c = att_c.masked_fill(padding_mask[:, None, None, :] == 0, float('-inf'))
        att_c = F.softmax(att_c, dim=-1)
        # att_c += F.softmax(self.weight_proj1(x), dim=1)[:, None, :, :] * att_c + F.softmax(self.weight_proj2(c), dim=1).squeeze()[:, None, None, :] * att_c
        att_c = self.attn_drop2(att_c)
        y_c = att_c @ v_c # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        y_c = y_c.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # cross gate strategy
        gate_self = F.sigmoid(self.proj_gate1(y))
        gate_c = F.sigmoid(self.proj_gate2(y_c))
        y = gate_self * y_c + gate_c * y
        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class MultiModalAttentionV3(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.qkv_x = nn.Linear(embed_dim, embed_dim*3)
        self.qkv_y = nn.Linear(embed_dim, embed_dim*3)

        self.cattn_x2y_drop = nn.Dropout(drop_out_rate)
        self.cattn_y2x_drop = nn.Dropout(drop_out_rate)
        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj_c = nn.Linear(embed_dim, embed_dim)
        self.gate_s = nn.Linear(embed_dim, embed_dim)
        self.gate_c = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v, self.cached_qkv_y = False, None, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v, self.cached_qkv_y = enable, None, None, None

    def forward(self, x, y, attn_x_mask=None):
        B, T, C = x.size()
        _, M, _ = y.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q_x, k_x, v_x = self.qkv_x(x).chunk(3, dim=-1)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.caching and self.cached_qkv_y is not None:
            q_y, k_y, v_y = self.cached_qkv_y
        else:
            q_y, k_y, v_y = self.cached_qkv_y = self.qkv_y(y).chunk(3, dim=-1)

        q_y = q_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        k_y = k_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v_y = v_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        # cross attention of x: (B, nh, T, hs) x (B, nh, M, hs) -> (B, nh, T, M)
        catt_x2y = (q_x @ k_y.transpose(-2, -1)) * (1.0 / math.sqrt(k_y.size(-1)))
        # cross attention score of x2y (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        cval_x2y = self.cattn_x2y_drop(F.softmax(catt_x2y, dim=-1)) @ v_y 
        cval_x2y = cval_x2y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # cross attention of y: (B, nh, M, hs) x (B, nh, T, hs) -> (B, nh, M, T)
        catt_y2x = (q_y @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
        # (B, nh, T, M) x (B, nh, M, T) -> (B, nh, T, T)
        catt_y2x = (catt_x2y @ catt_y2x) * (1.0 / math.sqrt(k_x.size(-1)))
        if attn_x_mask is not None:
            catt_y2x = catt_y2x.masked_fill(attn_x_mask == 0, float('-inf'))
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        cval_y2x = self.cattn_y2x_drop(F.softmax(catt_y2x, dim=-1)) @ v_x
        cval_y2x = cval_y2x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        cval = cval_x2y + cval_y2x
        # cval = self.resid_drop(self.proj_c(cval))+ x
        
        # key-value caching tech.
        if self.caching:
            if self.cached_k is None: 
                self.cached_k, self.cached_v = k_x, v_x
            else: 
                k_x = self.cached_k = torch.cat((self.cached_k, k_x), dim=2)
                v_x = self.cached_v = torch.cat((self.cached_v, v_x), dim=2)

        # self attention of x: (B, nh, T, hs) x (B, nh, T', hs) -> (B, nh, T, T'), T' is kv cached length
        satt = (q_x @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
        if attn_x_mask is not None:
            satt = satt.masked_fill(attn_x_mask == 0, float('-inf'))
        satt = F.softmax(satt, dim=-1)
        satt = self.attn_drop(satt)
        # self attention score
        sval = satt @ v_x # (B, nh, T, T') x (B, nh, T', hs) -> (B, nh, T, hs)
        sval = sval.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # gated fusion
        sgate = F.sigmoid(self.gate_s(sval))
        cgate = F.sigmoid(self.gate_c(cval))
        y = sgate * cval + cgate * sval
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class MultiModalAttentionV2(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.qkv_x = nn.Linear(embed_dim, embed_dim*3)
        self.qkv_y = nn.Linear(embed_dim, embed_dim*3)

        init_std = math.sqrt(1 / embed_dim / 3)
        w4x = torch.empty(embed_dim, 1)
        w4y = torch.empty(embed_dim, 1)
        w4xy = torch.empty(embed_dim, 1)
        nn.init.trunc_normal_(w4x, mean=0, std=init_std)
        nn.init.trunc_normal_(w4y, mean=0, std=init_std)
        nn.init.trunc_normal_(w4xy, mean=0, std=init_std)
        self.w4x = nn.Parameter(w4x, requires_grad=True)
        self.w4y = nn.Parameter(w4y, requires_grad=True)
        self.w4xy = nn.Parameter(w4xy, requires_grad=True)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.gate_s = nn.Linear(embed_dim, embed_dim)
        self.gate_c = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v, self.cached_qkv_y = False, None, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v, self.cached_qkv_y = enable, None, None, None

    def forward(self, x, y, attn_x_mask=None):
        B, T, C = x.size()
        _, M, _ = y.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q_x, k_x, v_x = self.qkv_x(x).chunk(3, dim=-1)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.caching and self.cached_k is not None:
            q_y, k_y, v_y = self.cached_qkv_y
        else:
            q_y, k_y, v_y = self.cached_qkv_y = self.qkv_y(y).chunk(3, dim=-1)

        q_y = q_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        k_y = k_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v_y = v_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        catt1 = (q_x @ self.w4x.view(1, self.n_head, C // self.n_head, 1)).expand([-1, -1, -1, M])
        catt2 = (q_y @ self.w4y.view(1, self.n_head, C // self.n_head, 1)).expand([-1, -1, -1, T]).transpose(-1, -2)
        catt3 = ((q_x * self.w4xy.view(1, self.n_head, 1, C // self.n_head)) @ k_y.transpose(-2, -1)) * (1.0 / math.sqrt(k_y.size(-1)))
        # cross attention of x: (B, nh, T, hs) x (B, nh, M, hs) -> (B, nh, T, M)
        catt = catt1 + catt2 + catt3
        catt_x2y, catt_y2x = F.softmax(catt, dim=-1), F.softmax(catt, dim=-2)
        catt_x2y, catt_y2x = self.attn_drop(catt_x2y), self.attn_drop(catt_y2x)
        # cross attention score
        cval_x2y = catt_x2y @ v_y # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        cval_x2y = cval_x2y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # (B, nh, T, M) x (B, nh, M, T) -> (B, nh, T, T)
        cval_y2x = catt_x2y @ catt_y2x.transpose(-1, -2) * (1.0 / math.sqrt(catt_y2x.size(-1)))
        if attn_x_mask is not None:
            cval_y2x = cval_y2x.masked_fill(attn_x_mask == 0, float('-inf'))
        cval_y2x = F.softmax(cval_y2x, dim=-1)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        cval_y2x = cval_y2x @ v_x
        cval_y2x = cval_y2x.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        cval = cval_x2y + cval_y2x + x
        # cval = cval_x2y + x
        
        # key-value caching tech.
        if self.caching:
            if self.cached_k is None: 
                self.cached_k, self.cached_v = k_x, v_x
            else: 
                k_x = self.cached_k = torch.cat((self.cached_k, k_x), dim=2)
                v_x = self.cached_v = torch.cat((self.cached_v, v_x), dim=2)

        # self attention of x: (B, nh, T, hs) x (B, nh, T', hs) -> (B, nh, T, T'), T' is kv cached length
        satt = (q_x @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
        if attn_x_mask is not None:
            satt = satt.masked_fill(attn_x_mask == 0, float('-inf'))
        satt = F.softmax(satt, dim=-1)
        satt = self.attn_drop(satt)
        # self attention score
        sval = satt @ v_x # (B, nh, T, T') x (B, nh, T', hs) -> (B, nh, T, hs)
        sval = sval.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # gated fusion
        sgate = F.sigmoid(self.gate_s(sval))
        cgate = F.sigmoid(self.gate_c(cval))
        y = sgate * cval + cgate * sval
        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class MultiModalAttention(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.qkv_x = nn.Linear(embed_dim, embed_dim*3)
        self.qkv_y = nn.Linear(embed_dim, embed_dim*3)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.gate_s = nn.Linear(embed_dim, embed_dim)
        self.gate_c = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.n_head = n_head

        # only used during inference
        self.caching, self.cached_k, self.cached_v, self.cached_qkv_y = False, None, None, None
    
    def kv_caching(self, enable: bool): self.caching, self.cached_k, self.cached_v, self.cached_qkv_y = enable, None, None, None

    def forward(self, x, y, attn_x_mask=None):
        B, T, C = x.size()
        _, M, _ = y.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q_x, k_x, v_x = self.qkv_x(x).chunk(3, dim=-1)
        q_x = q_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k_x = k_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v_x = v_x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # key-value caching tech.
        if self.caching:
            if self.cached_k is None: 
                self.cached_k, self.cached_v = k_x, v_x
                q_y, k_y, v_y = self.cached_qkv_y = self.qkv_y(y).chunk(3, dim=-1)
            else: 
                k_x = self.cached_k = torch.cat((self.cached_k, k_x), dim=2)
                v_x = self.cached_v = torch.cat((self.cached_v, v_x), dim=2)
                q_y, k_y, v_y = self.cached_qkv_y
        else:
            q_y, k_y, v_y = self.qkv_y(y).chunk(3, dim=-1)

        q_y = q_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        k_y = k_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)
        v_y = v_y.view(B, M, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, M, hs)

        # self attention of x: (B, nh, T, hs) x (B, nh, T', hs) -> (B, nh, T, T'), T' is kv cached length
        satt = (q_x @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
        # cross attention of y: (B, nh, M, hs) x (B, nh, T', hs) -> (B, nh, M, T')
        catt_y = (q_y @ k_x.transpose(-2, -1)) * (1.0 / math.sqrt(k_x.size(-1)))
        # fusion m2m-attn + t2m-attn: (B, nh, T, T') + 1/M x (B, nh, M, T') -> (B, nh, T, T')
        catt_y = torch.mean(F.softmax(catt_y, dim=-2) * catt_y, dim=-2, keepdim=True).repeat(1, 1, T, 1) 
        satt += catt_y
        if attn_x_mask is not None:
            satt = satt.masked_fill(attn_x_mask == 0, float('-inf'))
        satt = F.softmax(satt, dim=-1)
        satt = self.attn_drop(satt)
        # cross attention of x: (B, nh, T, hs) x (B, nh, M, hs) -> (B, nh, T, M)
        catt = (q_x @ k_y.transpose(-2, -1)) * (1.0 / math.sqrt(k_y.size(-1)))
        catt = F.softmax(catt, dim=-1)
        catt = self.attn_drop(catt)
        # self attention score
        sval = satt @ v_x # (B, nh, T, T') x (B, nh, T', hs) -> (B, nh, T, hs)
        sval = sval.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # cross attention score
        cval = catt @ v_y # (B, nh, T, M) x (B, nh, M, hs) -> (B, nh, T, hs)
        cval = cval.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # gated fusion
        sgate = F.sigmoid(self.gate_s(sval))
        cgate = F.sigmoid(self.gate_c(cval))
        y = sgate * cval + cgate * sval
        # output projection
        y = self.resid_drop(self.proj(y))
        return y


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class BlockPro(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, ff_size=1024):
        super().__init__()
        self.ln0 = RMSNorm(embed_dim)
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.attn = MultiModalAttentionV3(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            SwiGLU(embed_dim, ff_size),
            nn.Dropout(drop_out_rate),
        )

    def kv_caching(self, enable: bool):
        self.attn.kv_caching(enable)

    def forward(self, x, y, attn_mask=None):
        x = x + self.attn(self.ln0(x), self.ln1(y), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    


class Block(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, ff_size=1024):
        super().__init__()
        self.ln0 = RMSNorm(embed_dim)
        self.ln0_t = RMSNorm(embed_dim)
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.attn0 = CrossAttention(embed_dim, n_head, drop_out_rate)
        self.attn = SelfAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            SwiGLU(embed_dim, ff_size),
            nn.Dropout(drop_out_rate),
        )

    def kv_caching(self, enable: bool):
        self.attn0.kv_caching(enable)
        self.attn.kv_caching(enable)

    def forward(self, x, y, freqs_cis, attn_mask=None):
        x = x + self.attn0(self.ln0(x), self.ln0_t(y))
        x = x + self.attn(self.ln1(x), freqs_cis, attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
    

class MultiModalBlock(nn.Module):

    def __init__(self, embed_dim=512, n_head=8, drop_out_rate=0.1, ff_size=1024):
        super().__init__()
        self.ln0 = RMSNorm(embed_dim)
        self.ln0_t = RMSNorm(embed_dim)
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)
        self.ln2_t = RMSNorm(embed_dim)
        self.attn0 = CrossAttentionPro(embed_dim, n_head, drop_out_rate)
        self.attn = SelfAttention(embed_dim, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            SwiGLU(embed_dim, ff_size),
            nn.Dropout(drop_out_rate),
        )

    def kv_caching(self, enable: bool):
        self.attn0.kv_caching(enable)
        self.attn.kv_caching(enable)

    def forward(self, x, y, attn_mask=None):
        x = x + self.attn0(self.ln0(x), self.ln0_t(y), attn_mask)
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
