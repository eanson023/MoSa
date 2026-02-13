import os
from os.path import join as pjoin

import random
import torch

from models.transformer.transformer import Transformer
from models.tokenizer.tokenizer import SQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.tools import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical
from functools import partial

from utils.motion_process import recover_from_ric


import numpy as np
clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    vq_model = SQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.scales,
                vq_opt.nb_code_st,
                vq_opt.nb_code_ed,
                vq_opt.code_dim,
                vq_opt.width,
                vq_opt.width_mul,
                vq_opt.depth,
                vq_opt.slot_group,
                )
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.vq_name}')
    return vq_model

def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = Transformer(scales = model_opt.scales,
                                      nb_code_st = model_opt.nb_code_st,
                                      nb_code_ed = model_opt.nb_code_ed,
                                      code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      mm_attn=model_opt.mm_attn,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

@torch.no_grad()
def text2bvh(t2m_transformer, vq_model, converter, cached_dir, uid, text, step, up2motion_len, inv_transform, motion_length=196, seed=10107, repeat_times=1,
             cond_scale=4, top_k=0.8, top_p=0.9, temperature=1.0, nb_joints=22):
    device = next(t2m_transformer.parameters()).device
    prompt_list = [text]
    length_list = [motion_length]
    
    token_lens = torch.LongTensor(length_list)
    token_lens[token_lens > 196] = 196
    token_lens = token_lens // 4
    token_lens = token_lens.to(device).long()

    m_length = token_lens * 4
    captions = prompt_list

    for r in range(repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            z_hat = t2m_transformer.generate_at_scale(step, captions, m_length,
                                            cond_scale=cond_scale,
                                            vq_model=vq_model,
                                            top_k=top_k, 
                                            top_p=top_p, 
                                            temperature=temperature,
                                            up2motion_len = up2motion_len,
                                            seed=seed
                                            )
            
            pred_motions = vq_model.forward_decoder(z_hat)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(cached_dir, f'{uid}')

            os.makedirs(animation_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), nb_joints).numpy()
            

            bvh_path = pjoin(animation_path, "sample_repeat%d.bvh" % (r))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

        return bvh_path, joint, len(joint_data)



if __name__ == "__main__":

    text = 'a person walk forward'
    motion_len = 180
    uid = 12138

    parser = EvalT2MOptions()
    opt = parser.parse()
    # fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 263
    opt.nb_joints = 22

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)


    #######################
    ######Loading SVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model = load_vq_model(vq_opt)

    model_opt.scales = vq_opt.scales
    model_opt.nb_code_st = vq_opt.nb_code_st
    model_opt.nb_code_ed = vq_opt.nb_code_ed
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'net_best_fid.tar')

    t2m_transformer.eval()
    vq_model.eval()

    t2m_transformer.to(opt.device)
    vq_model.to(opt.device) 

    ##### ---- Dataloader ---- #####

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean
    
    converter = Joint2BVHConvertor()
    
    text2bvh = partial(text2bvh, converter=converter, inv_transform=inv_transform)
    bvh = text2bvh(t2m_transformer, vq_model, text, uid, motion_length=motion_len)
    print(bvh)
    
    