import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import clip
import numpy as np
from functools import partial
from utils.tools import cal_performance_naive, eval_decorator, sample_with_top_k_top_p_
from models.transformer.network import Block, MultiModalBlock
from models.transformer.embed_rope import compute_cis

class CLIPModelWrapper(nn.Module):
    def __init__(self, clip_version):
        super(CLIPModelWrapper, self).__init__()
        self.load_and_freeze_clip(clip_version)

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        # clip.model.convert_weights(
        #     clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        print("***clip model loaded***")
        self.clip_model = clip_model

    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)

        # self.clip_model.encode_text
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        # sentence level and word level
        word_mask = text > 0
        # max_len = max(word_mask.sum(dim=-1))
        # x = x[:, :max_len]
        # word_mask = word_mask[:, :max_len]
        return sent_x.float(), x.float(), word_mask


class Transformer(nn.Module):
    def __init__(self, scales, nb_code_st, nb_code_ed, code_dim, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None, mm_attn=True,
                 **kargs):
        super(Transformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')
        # 0. hyperparameters
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt

        self.pad_id = -1

        self.cond_drop_prob = cond_drop_prob
        
        self.scales = scales
        self.first_t = self.scales[0]
        self.num_stages_minus1 = len(self.scales)-1
        self.rng = torch.Generator()

        # 1. input (word) embedding
        self.input_process = nn.Linear(self.code_dim, self.latent_dim)
        self.input_emb = nn.Linear(self.clip_dim, self.latent_dim)
        
        # 2. start embedding
        self.T = sum(s for s in scales)
        init_std = math.sqrt(1 / latent_dim / 3)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_t, latent_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. motion length embedding
        self.len_embed = nn.Embedding(self.scales[-1], latent_dim)
        nn.init.trunc_normal_(self.len_embed.weight.data, mean=0, std=init_std)

        # 4. absolute position embedding
        # pos_1LC = []
        # for i, s in enumerate(self.scales):
        #     pe = torch.empty(1, s, latent_dim)
        #     nn.init.trunc_normal_(pe, mean=0, std=init_std)
        #     pos_1LC.append(pe)
        # pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, T, embed_dim
        # assert tuple(pos_1LC.shape) == (1, self.T, latent_dim)
        # self.pos_embedding = nn.Parameter(pos_1LC)
        scale_freqs_cis=[]
        self.rope_norm = latent_dim // num_heads
        self.compute_cis = partial(compute_cis, dim=latent_dim//num_heads, theta=100, normalize=self.rope_norm)
        for i, pn in enumerate(self.scales):
            freqs_cis = self.compute_cis(end_x = pn)
            scale_freqs_cis.append(freqs_cis)
        self.register_buffer('freqs_cis', torch.cat(scale_freqs_cis,dim=0))#(L,latent_dim//head)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.scales), latent_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 5. backbone blocks
        attn_class = MultiModalBlock if mm_attn else Block
        self.blocks = nn.ModuleList([attn_class(self.latent_dim, num_heads, dropout, ff_size) for _ in range(num_layers)])

        # 6. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        lvl_1L: torch.Tensor = torch.cat([torch.full((s, ), i) for i, s in enumerate(self.scales)]).view(1, self.T)
        self.register_buffer('lvl_1L', lvl_1L)

        d: torch.Tensor = torch.cat([torch.full((s, ), i) for i, s in enumerate(self.scales)]).view(1, self.T, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        attn_bias_for_masking = torch.where(d >= dT, 1., 0.).reshape(1, 1, self.T, self.T)
        self.register_buffer('attn_mask', attn_bias_for_masking.contiguous())

        # 7. classifier head
        # self.head_nm = AdaLNBeforeHead(self.latent_dim, self.clip_dim)
        nb_codes = [round(nb_code) for nb_code in np.linspace(nb_code_st, nb_code_ed, len(self.scales))]
        self.heads = nn.ModuleList([nn.Linear(self.latent_dim, nb_codes[i]) for i in range(len(self.scales))])

        # 8. init networks
        self.apply(self.__init_weights)

        # 9. loading clip
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = CLIPModelWrapper(self.clip_version)


    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def cond_mask(self, cond, force_mask=False):
        bs =  cond.shape[0]
        if force_mask:
            return torch.zeros(bs, device=cond.device)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob)
            return (1. - mask)
        else:
            return torch.ones(bs, device=cond.device)

    @torch.no_grad()
    @eval_decorator
    def generate_at_scale(self, q,
                 conds,
                 m_lens,
                 cond_scale: int,
                 vq_model,
                 top_k = 0.8,
                 top_p = 0.95,
                 temperature = 1.0,
                 seed=None,
                 up2motion_len=False):

        device = next(self.parameters()).device
        bs = len(m_lens)
        if seed is None:
            rng = None
        else:
            rng = torch.Generator(device)
            rng.manual_seed(seed)

        assert q < len(self.scales)

        '''
        Preparing input
        '''
        sent_vector, word_vector, word_mask = self.clip_model(conds) # [b, t], [b, t, 512]
        len_embed = self.len_embed((m_lens // self.opt.unit_length)-1).repeat(2, 1)
        # classifier-free guidance
        cond_mask = self.cond_mask(word_vector, force_mask=True)
        word_vector = torch.cat((word_vector, word_vector * cond_mask[:, None, None]), dim=0)
        cond = self.input_emb(word_vector)
        cond += len_embed.unsqueeze(1)

        sos = torch.zeros(bs*2, self.first_t, self.latent_dim).float().to(device)
        sos += self.pos_start.expand(bs * 2, self.first_t, -1)
        # sent_vector = torch.cat((sent_vector, sent_vector * cond_mask[:, None]), dim=0)
        # sos += self.input_emb(sent_vector).unsqueeze(1)
        sos += len_embed.unsqueeze(1)

        pos_embd = self.lvl_embed(self.lvl_1L) # + self.pos_embedding

        next_token_map = sos + pos_embd[:, :self.first_t]

        scales = self.scales[:q+1]
        num_stages_minus1 = len(scales) - 1
        
        if up2motion_len:
            z_hat = sos.new_zeros(bs, self.code_dim, self.scales[-1])
        else:
            z_hat = sos.new_zeros(bs, self.code_dim, scales[-1])

        for b in self.blocks: b.kv_caching(True)
        
        cur_T = 0
        for q, pn in enumerate(scales):
            ratio = q / max(num_stages_minus1, 1e-4)
            xseq = next_token_map
            cur_T += pn
            freqs_cis_cur = self.freqs_cis[cur_T-pn: cur_T,:]
            # (b, num_token, seqlen)
            for block in self.blocks:
                xseq = block(xseq, cond, freqs_cis_cur)
            
            # logits = self.heads[s](self.head_nm(xseq.float(), sent_vector_bf).float()).float().permute(0, 2,1)
            logits = self.heads[q](xseq.float()).float().permute(0, 2,1)
            # classifier-free guidance ratio
            t = cond_scale * ratio
            logits = (1+t) * logits[:bs] - t * logits[bs:]

            logits = logits.permute(0, 2, 1).contiguous()  # (b, seqlen, ntoken)
            # _, pred_ids = torch.topk(F.softmax(logits, dim=-1), k=1, dim=-1)
            # pred_ids = pred_ids[:, :, 0]
            # print(logits.shape, self.opt.num_tokens)
            pred_ids = sample_with_top_k_top_p_(logits, rng=rng, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1)[:, :, 0]

            '''
            Preparing next token input
            '''
            z_BCs =  F.embedding(pred_ids, vq_model.quantizer.layers[q].codebook).permute(0, 2, 1).contiguous()
            z_hat, next_token_map = vq_model.quantizer.get_next_autoregressive_input(q, z_hat, z_BCs, scales)
            if q < num_stages_minus1:   # prepare for next stage
                # NOTE fix bug: don't use view!!!
                # next_token_map = next_token_map.view(bs, self.code_dim, -1).transpose(1, 2)
                next_token_map = self.input_process(next_token_map) + pos_embd[:, cur_T:cur_T + scales[q+1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.kv_caching(False)
        return z_hat
    