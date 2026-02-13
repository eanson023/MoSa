import os
from os.path import join as pjoin

import clip
import torch
from torch.utils.data import DataLoader, RandomSampler

from models.tokenizer.tokenizer import SQVAE
from models.tokenizer.vq_trainer import VQTokenizerTrainer
from options.vq_option import arg_parse
from data.t2m_dataset import MotionDataset
from utils import paramUtil
import numpy as np

from models.t2m_eval_wrapper import EvaluatorModelWrapper
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

os.environ["OMP_NUM_THREADS"] = "1"

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/HumanML3D/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        fps = 20
        radius = 4
        opt.unit_length = 2**len(opt.width_mul)
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.unit_length = 2**len(opt.width_mul)
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    elif opt.dataset_name == "motionx":
        opt.data_root = './dataset/Motion-X/'
        opt.motion_dir = pjoin(opt.data_root, 'vector_263')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        dim_pose = 263
        fps = 20
        radius = 4
        opt.unit_length = 2**len(opt.width_mul)
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/motionx/Comp_v6_KLD005/opt.txt'
        mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
        std = np.load(pjoin(opt.data_root, 'Std.npy'))
        train_split_file = pjoin(opt.data_root, 'train.txt')
        val_split_file = pjoin(opt.data_root, 'val.txt')
    else:
        raise KeyError('Dataset Does not Exists')

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    net = SQVAE(opt,
                # clip_model,
                dim_pose,
                opt.scales,
                opt.nb_code_st,
                opt.nb_code_ed,
                opt.code_dim,
                opt.width,
                opt.width_mul,
                opt.depth,
                opt.slot_group,
                )

    pc_vq = sum(param.numel() for param in net.parameters() if param.requires_grad)
    print(net)
    
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    trainer = VQTokenizerTrainer(opt, vq_model=net)

    if opt.dataset_name == "t2m":
        train_dataset = MotionDataset(opt, mean, std, train_split_file)
        val_dataset = MotionDataset(opt, mean, std, val_split_file)
        batch_size = opt.batch_size
        num_workers = 4
        train_loader_ft = None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                                    pin_memory=True, shuffle=True)
    else:
        train_dataset = MotionDataset(opt, mean, std, train_split_file, part=['short'])
        val_dataset = MotionDataset(opt, mean, std, val_split_file, part=['short'])
        # Using long-duration motion data as OOD data.
        # `feat_bias=False` the mean and std are soft references; they do not need to perform 
        # feat bias augmentation since it has already been done beforehand.
        train_dataset_ft = MotionDataset(opt, mean, std, train_split_file, part=['long'], feat_bias=False)
        batch_size = opt.batch_size // 2
        num_workers = 2
        train_loader_ft = DataLoader(train_dataset_ft, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                              pin_memory=True, sampler=RandomSampler(train_dataset_ft, num_samples=min(len(train_dataset_ft), len(train_dataset))))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_workers,
                                    pin_memory=True, sampler=RandomSampler(train_dataset, num_samples=min(len(train_dataset_ft), len(train_dataset))))
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=True, pin_memory=True)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)
    trainer.train(train_loader, train_loader_ft, val_loader, eval_val_loader, eval_wrapper, None)
