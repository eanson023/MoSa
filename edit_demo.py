import os
from os.path import join as pjoin

import time
import torch

from models.transformer.transformer import Transformer
from models.tokenizer.tokenizer import SQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.tools import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain, kit_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    vq_model = SQVAE(vq_opt,
                dim_pose,
                vq_opt.scales,
                vq_opt.nb_code_st,
                vq_opt.nb_code_ed,
                vq_opt.code_dim,
                vq_opt.width,
                vq_opt.width_mul,
                vq_opt.depth,
                vq_opt.slot_group,
                )
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_mm.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {opt.vq_name}')
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


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    # fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./editing', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)

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
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean
    
    motion = np.load(opt.source_motion)
    m_length = len(motion)
    print(m_length)
    print('-----------------------')
    motion = (motion - mean) / std
    if model_opt.max_motion_length > m_length:
        motion = np.concatenate([motion, np.zeros((model_opt.max_motion_length - m_length, motion.shape[1])) ], axis=0)
    motion = torch.from_numpy(motion)[None].to(opt.device)

    prompt_list = []
    length_list = []

    if opt.text_prompt != "":
        prompt_list.append(opt.text_prompt)
        # length_list.append(opt.motion_length)
        length_list.append(m_length)
    elif opt.text_path != "":
        with open(opt.text_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                infos = line.split('#')
                prompt_list.append(infos[0])
                if len(infos) == 1 or (not infos[1].isdigit()):
                    raise "Please input a motion length like 'a persion walk forward.#196'!!!"
                else:
                    length_list.append(int(infos[-1]))
    else:
        raise "A text prompt, or a file a text prompts are required!!!"
    # print('loading checkpoint {}'.format(file))
    
    token_lens = torch.LongTensor(length_list)
    token_lens[token_lens>model_opt.max_motion_length] = model_opt.max_motion_length
    token_lens = token_lens // 4
    token_lens = token_lens.to(opt.device).long()

    m_length = token_lens * 4
    captions = prompt_list
    print_captions = captions[0]
    
    _edit_slice = opt.mask_edit_section
    edit_slice = []
    for eds in _edit_slice:
        _start, _end = eds.split(',')
        _start = eval(_start)
        _end = eval(_end)
        edit_slice.append([_start, _end])
    
    with torch.no_grad():
        tokens = vq_model.quantize(motion, m_length)
        
    ### build editing mask, TOEDIT marked as 1 ###
    edit_mask = torch.zeros_like(tokens[-1])
    # seq_len = edit_mask.shape[1]
    seq_len = token_lens
    total_length = 0
    for _start, _end in edit_slice:
        if isinstance(_start, float):
            _start = int(_start*seq_len)
            _end = int(_end*seq_len)
        else:
            _start //= 4
            _end //= 4
        edit_mask[:, _start: _end] = 1
        total_length += (_end - _start)
        print_captions = f'{print_captions} [{_start*4/20.}s - {_end*4/20.}s]'
    
    edit_mask = edit_mask.float()

    # target region motion_length
    token_lens = torch.LongTensor([total_length * 4])
    token_lens[token_lens>model_opt.max_motion_length] = model_opt.max_motion_length
    token_lens = token_lens // 4
    token_lens = token_lens.to(opt.device).long()

    region_length = token_lens * 4
    # import pdb; pdb.set_trace()

    sample = 0
    kinematic_chain = kit_kinematic_chain if opt.dataset_name == 'kit' else t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    total_time = time.time()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():
            z_hat = t2m_transformer.edit(captions, 
                                         tokens, 
                                         edit_mask,
                                         m_length,
                                         cond_scale=opt.cond_scale,
                                         vq_model=vq_model,
                                         top_k=opt.top_k, 
                                         top_p=opt.top_p, 
                                         temperature=opt.temperature,
                                         more_smooth=True,
                                         )
            
            pred_motions = vq_model.forward_decoder(z_hat)

            pred_motions = pred_motions.detach().cpu().numpy()
            source_motions = motion.detach().cpu().numpy()

            data = inv_transform(pred_motions)
            source_data = inv_transform(source_motions)
        
        # data = np.load('dataset/Motion-X/vector_263/dance/subset_0002/Several_Methods_Of_Ballet_Idling_clip_5.npy')[None, :, :]

        for k, (caption, joint_data, source_data)  in enumerate(zip(captions, data, source_data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.nb_joints).numpy()

            source_data = source_data[:m_length[k]]
            soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), 22).numpy()

            if opt.nb_joints == 21:
                save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
                plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20, radius=246 * 12)
            else:
                # bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
                # _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

                bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
                _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

                if r==0:
                    bvh_path = pjoin(animation_path, "sample%d_source_len%d.bvh" % (k, m_length[k]))
                    _, _ = converter.convert(soucre_joint, filename=bvh_path, iterations=100, foot_ik=False)


                save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
                ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))
                source_save_path = pjoin(animation_path, "sample%d_source_len%d.mp4"%(k, m_length[k]))

                # plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=print_captions, fps=20)
                plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=20)
                if r==0:
                    plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='None', fps=20)
                np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
                # np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), ik_joint)

    total_time = time.time() - total_time
    print(f'Average Inference Time: {total_time/opt.repeat_times:.5f}')


# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp3 -msec 0.4,0.7 --text_prompt "A man picks something from the ground using his right hand."
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp3 -msec 0.4,0.7 --text_prompt "A man picks something from the ground using his right hand." --source_motion "dataset/HumanML3D/new_joint_vecs/002198.npy"
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp3 -msec 0.3,0.7 --text_prompt "A person is walking while raise hands" --source_motion "dataset/HumanML3D/new_joint_vecs/002198.npy"  --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp0 -msec 0.3,0.7 --text_prompt "person is working on their boxing form" --source_motion "dataset/HumanML3D/new_joint_vecs/000933.npy"  --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp0 -msec 0.4,0.7 --text_prompt "person is working on their boxing form" --source_motion "dataset/HumanML3D/new_joint_vecs/002198.npy"  --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp18_2 -msec 0.0,0.5 --text_prompt "a man walks in a zig zag pattern." --source_motion "dataset/HumanML3D/new_joint_vecs/006591.npy" --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp21 -msec 0.0,0.2 0.75,1.0 --text_prompt "a person does a jumping jack" --source_motion "dataset/HumanML3D/new_joint_vecs/006591.npy" --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp22 -msec 0.0,0.2 0.8,1.0 --text_prompt "someone is walking diagonally across the screen" --source_motion "dataset/HumanML3D/new_joint_vecs/008642.npy" --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp23 -msec 0.0,0.5 --text_prompt "a person squats down and stands up." --source_motion "dataset/HumanML3D/new_joint_vecs/000926.npy" --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp24 -msec 0.0,0.5 --text_prompt "a person is making a high kick with his left leg." --source_motion "dataset/HumanML3D/new_joint_vecs/001973.npy" --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp25 -msec 0.6,1.0 --text_prompt "a person is sitting down on the ground." --source_motion "dataset/HumanML3D/new_joint_vecs/007846.npy" --repeat_times 10
# python edit_demo.py --name t2m_pkeep_ope_ffsize768_bs64_milestone100_200_val --gpu_id 0 --ext exp26 -msec 0.0,0.1 --text_prompt "a person is sitting down on the ground." --source_motion "dataset/HumanML3D/new_joint_vecs/000926.npy" --repeat_times 10
