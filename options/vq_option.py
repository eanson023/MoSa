import argparse
import os
import torch

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit', 'motionx'], help='dataset directory')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')

    ## optimization
    parser.add_argument('--max_epoch', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[300], nargs="+", type=int, help="learning rate schedule (epoch)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--using_znorm", action="store_true", help=', transforming the Euclidean distance into cosine similarity')
    parser.add_argument("--code_dim", type=int, default=512, help="codebook dimension C")
    parser.add_argument("--scales", type=str, default="3_6_10_15_20_25_30_36_42_49", help="scalble quantizer scales, represents a predefined scheduler moving from coarse to fine")
    parser.add_argument("--nb_code_st", type=int, default=256, help="nb of embedding start")
    parser.add_argument("--nb_code_ed", type=int, default=768, help="nb of embedding end")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--width", type=int, default=256, help="width of the network")
    parser.add_argument("--width_mul", type=str, default="1_2_1", help="width mul rate of the network, the length denotes encoder-decoder down-upsampling rate")
    parser.add_argument("--depth", type=int, default=2, help="num of resblocks for each res")
    parser.add_argument('--slot_group', type=int, default=10, help='virtual motion slot group num')

    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--tiny', action="store_true", help="training on small datasets, small model")
    parser.add_argument('--phi_k', type=int, default=3, help='conv block phi kernel')
    parser.add_argument('--phi_depth', type=int, default=2, help='conv block depth')

    parser.add_argument('--commit_beta', type=float, default=1.0)

    ## other
    parser.add_argument('--name', type=str, default="svq_nq10_nc256_768_noshare_phik3_phidepth2", help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", help='Name of this trial')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_every', default=50, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=48, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_e', default=1, type=int, help='save eval results every n epoch')
    parser.add_argument('--eval_every_i', default=2000, type=int, help='save eval results every n iters')
    parser.add_argument('--eval_start_e', type=int, default=200, help='Frequency of animating eval results, (epoch)')
    # parser.add_argument('--early_stop_e', default=5, type=int, help='early stopping epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='Layers of GRU')

    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')
    parser.add_argument('--ext', type=str, default='default', help='eval file prefix')

    ## For Res Predictor only
    parser.add_argument('--vq_name', type=str, default="svq_nq10_nc256_768_noshare_phik3_phidepth", help='Name of this trial')
 
    parser.add_argument("--seed", default=2025, type=int)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    if opt.tiny:
        opt.width_mul = "1_1_1"
        opt.depth = 0
        opt.phi_depth = 0
        opt.scales = "3_25_49"
    opt.scales = tuple(map(int, opt.scales.replace('-', '_').split('_')))
    opt.width_mul = tuple(map(int, opt.width_mul.replace('-', '_').split('_')))

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt