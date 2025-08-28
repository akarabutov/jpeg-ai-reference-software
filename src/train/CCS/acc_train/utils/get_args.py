# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
from src.codec.components import DecoderFactory, HyperScaleDecoderFactory, EncoderFactory, HyperDecoderFactory, HyperEncoderFactory

__all__ = ['get_args']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args(args_list = None):
    parser = argparse.ArgumentParser(description='Deep Image Compression Trainer.')
    # using modelarts for training
    parser.add_argument('--overfit', default=False, action='store_true', help='Make overfitting')
    parser.add_argument('--tar_file', type=str, default='', help='')
    parser.add_argument('--data_url', type=str, default='', help='')
    parser.add_argument('--rec_dir', type=str, default='', help=r'Dirtectory for storing reconstructed signals')
    parser.add_argument('--use_scc_dataset', type=str2bool, default=False, help='Use scc dataset or not, defalult True')
    parser.add_argument('--cfg_path', type=str, nargs="+", default=['tools_off.json', 'profiles/base.json'], help= '')
    parser.add_argument('--val_tar_file', type=str, default='', help='')
    parser.add_argument('--gpu_id', type=str, default='0', help='')
    parser.add_argument('--train_cfg_json', type=str, default='cfg/train.json', help='A path to a json file with desciption of training')
    parser.add_argument('--train_stages_json', type=str, default='cfg/train_stages.json', help='A path to a json file with desciption of training stages')
    parser.add_argument('--cuda_empty_cash_each_batch',
                        type=int,
                        default=0,
                        help='make cuda.empty_cash each train/val batch')
    parser.add_argument(
        '--copy_to_train_url_dir',
        type=str,
        default='',
        help=
        'This directory will be copied to temporary directory used for training. Can be used for resuming from other training (copying best.pth)'
    )
    parser.add_argument('--resume_from_stage',
                        type=str,
                        default=None,
                        help='try to resume from best.pth of this stage')
    # parameters of dataset
    parser.add_argument('--lst', type=str, default='', help='')
    parser.add_argument('--data_dir', type=str, default='', help='')
    parser.add_argument('--train_url', type=str, default='', help='')
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument(
        '--resume_opt',
        type=str2bool,
        default=False,
        help=
        'if true, both optimizer and weights are loaded. if 0 only weights of args.resume is loaded'
    )
    parser.add_argument(
        '--collect_only',
        type=str2bool,
        default=False,
        help=
        'if true, data collection process will be executed after training'
    )
    parser.add_argument(
        '--automatic_resume_on_crash',
        type=str2bool,
        default=True,
        help=
        'if true, training process of a worker will be restarted if it crashes. will resume from latest epoch in this case'
    )
    parser.add_argument('--val_lst', type=str, default='', help='')
    parser.add_argument('--val_data_dir', type=str, default='', help='')
    parser.add_argument('--test_data_dir',
                        type=str,
                        default=os.path.join(os.getcwd(), 'data', 'test'),
                        help='')
    parser.add_argument(
        '--use_automatic_testing',
        type=str2bool,
        default=False,
        help=
        'if true, inference test on data in `test_data_dir` will be done every `automatic_testing_epoch_period` epochs. '
    )
    parser.add_argument(
        '--use_automatic_testing_best',
        type=str2bool,
        default=False,
        help=
        'if true, inference test of best models on data in `test_data_dir` will be done every `automatic_testing_epoch_period` epochs. '
    )    
    parser.add_argument(
        '--generate_test_summary',
        type=str2bool,
        default=False,
        help=
        'if inference test is performed, generate summary file (generate_test_summary == True). '
    )    
    parser.add_argument('--automatic_testing_epoch_period',
                        type=int,
                        default=4,
                        help='how often to perform automatic tests if enabled')
    # training parameters
    parser.add_argument('--sym_flag',
                        type=str2bool,
                        default=True,
                        help='SGMM -> true, GMM -> false, please keep it be ture ')
    parser.add_argument('--N', type=int, default=160, help='')
    parser.add_argument('--N_UV', type=int, default=96, help='')
    parser.add_argument('--beta_list', type=str, default='', help='')
    parser.add_argument('--N_G', type=int, default=3, help='use 3 or 2 for SGMM3 or SGMM2')
    parser.add_argument('--beta', type=float, default=0.002, help='')
    parser.add_argument('--workers', type=int, default=10, help='')
    parser.add_argument('--epochs', type=int, default=64, help='')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--crop_size', type=int, default=320, help='')
    parser.add_argument('--crop_number', type=int, default=1, help='')
    parser.add_argument('--print_freq', type=int, default=100, help='')
    parser.add_argument('--scale_bound', type=float, default=1e-9, help='')
    parser.add_argument('--entropy',
                        type=str,
                        choices=['gaussian', 'laplacian'],
                        default='gaussian',
                        help='')
    parser.add_argument('--loss_type',
                        type=str,
                        choices=['mse', 'msssim', 'mix'],
                        default='mse',
                        help='')
    parser.add_argument('--msssim_weight', type=float, default=0.5, help='')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='')
    parser.add_argument('--loss_weights', type=str, default='8,1;1,8;8,1', help=r'Weights in the following format: "dec0_enc0,dec0_enc1;dec1_enc0,dec1_enc1;dec2_enc0,dec2_enc1"')
    parser.add_argument('--loss_factors', type=str, default='0.5_0.5_0.5', help=r'Weights in the following format: "factorY_factorCb_factorCr"')
    parser.add_argument('--l1', type=float, default=5e-9, help='')
    parser.add_argument('--enable_gvae',
                        type=int,
                        default=0,
                        help='enable the gvae in stageII-stageIV training')

    # parameters of optimazition
    parser.add_argument('--opt_type',
                        default='adam',
                        choices=['adam'],
                        help='')
    parser.add_argument('--lr_type',
                        type=str,
                        default='warmup_anneal.step',
                        choices=[
                            'warmup_step.step', 'warmup_anneal.step', 'step.epoch',
                            'reduce_on_plateau.epoch'
                        ],
                        help='')
    parser.add_argument('--base_warmup_epoch', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--anneal_final_lr',
                        type=float,
                        default=None,
                        help='only for warmup_anneal.step, the value of LR after beging annealed')
    parser.add_argument('--lr_steps',
                        type=str,
                        default='20,30,40',
                        help='Numbers of epochs when learning rate will be decreased.')
    parser.add_argument('--adam_lr_type',
                        type=str,
                        default='warmup_anneal.step',
                        choices=[
                            'warmup_step.step', 'warmup_anneal.step', 'step.epoch',
                            'reduce_on_plateau.epoch'
                        ],
                        help='')

    parser.add_argument('--patience',
                        type=int,
                        default=5,
                        help='Patience for reduce on plateu scheduler')
    parser.add_argument('--wd', type=float, default=0, help='')
    parser.add_argument('--factor',
                        type=float,
                        default=0.5,
                        help='Factor for reduce on plateu scheduler')
    parser.add_argument('--skip_thre',
                        type=float,
                        default=0.0,
                        help='skip threhold in train, 0.0 is recommended')
    parser.add_argument('--cube_flag_thre',
                        type=float,
                        default=1.0,
                        help='skip cube flag threhold in train, 0.0/1.0 are recommended for high/base OP')
    parser.add_argument('--seed', type=int, default=10, help='')

    parser.add_argument(
        '--max_pic_area_in_validation',
        type=int,
        default=40000 * 60000,
        help='pictures with greater area will be skipped in validation (to avoid OOM problem)')

    # parameters for index of top model
    parser.add_argument('--best_n', type=int, default=4, help='')

    # 420 or 444
    parser.add_argument('--color', type=str, choices=['420', '444'], default='420', help='')
    # boudanry handling
    parser.add_argument('--bh', type=int, default='1', help='')

    # set a bigger bottleneck for high rate model
    parser.add_argument('--N_Y', type=int, default=0, help='')

    # set type of decoder
    parser.add_argument('--vae_encoder_type_list',
                        type=str,
                        default=["bop", "hop"],
                        nargs="+",
                        #choices=EncoderFactory().keys(),
                        help='List with type of encoders')
    parser.add_argument('--vae_decoder_type_list',
                        type=str,
                        default=["bop", "hop", "sop"],
                        nargs="+",
                        #choices=DecoderFactory().keys(),
                        help='List with type of decoders')
    parser.add_argument('--hyper_decoder_type',
                        type=str,
                        default='basic',
                        choices=HyperDecoderFactory().keys(),
                        help='type of hyper_decoder')

    parser.add_argument('--hyper_scale_decoder_type',
                        type=str,
                        default='hsd',
                        choices=HyperScaleDecoderFactory().keys(),
                        help='type of hyper_scale_decoder')
    parser.add_argument('--hyper_encoder_type',
                        type=str,
                        default='basic',
                        choices=HyperEncoderFactory().keys(),
                        help='type of hyper_encoder')

    # use Context, Gather and GMM or not
    parser.add_argument( '--sigma_quant_level', type=int, default=35, help='quantization levels of sigma')
    parser.add_argument( '--sigma_quant_max', type=float, default=100, help='quantization max of sigma')
    parser.add_argument( '--sigma_quant_min', type=float, default=0.11, help='quantization max of sigma')
    parser.add_argument( '--sigma_bound_offset', type=float, default=0.5, help='boundary offset of sigma')

    # if use tarfile on cloud
    parser.add_argument('--cloud_tar',
                        type=str2bool,
                        default=False,
                        help=' in rss it should be False')

    # set save epochs number for different data set
    parser.add_argument('--save_epoch',
                        type=int,
                        default='1',
                        help='for COCO dataset use 1, for Jpegai use 22 ')
    parser.add_argument('--frozen_part',
                        type=str,
                        nargs="+",
                        default=[],
                        choices=['entropy', 'synthesis', 'gain_unit', 'analysis'],
                        help='Froze parts of networks')
    parser.add_argument('--cal_entropy_on_val',
                        type=str2bool,
                        default=False,
                        help=' calulate the channel wise entropy of residual tensor')
    # parser.add_argument('--reduce_lr_on_plateau', type=int, default=1, help='Use ReduceLROnPlateau scheduler')
    # args AMP, DDP and MDP
    # parser.add_argument('--local_rank', type=int, default=-1, help='local rank for DDP or MDP')
    parser.add_argument('--amp', type=str2bool, default=True, help='Use amp or not, defalult True')
    parser.add_argument('--zero_redundancy_optimizer', type=str2bool, default=False, help='')
    # parser.add_argument('--zero_redundancy_optimizer', type=str2bool, default=True, help='')
    # TODO: add MDP
    args, unparsed = parser.parse_known_args(args_list)
    # args.mem_format = 'channel_last' if args.amp else 'channel_first'
    args.mem_format = 'channel_first'
    # TODO: debug unconvergence problem when amp and channe_last are used, and zero_redundancy_optimizer set as false
    # when amp, channe_last and zero_redundancy_optimizer are used, the unconvergence problem gone. find out the reason.
    assert args.mem_format == 'channel_first'
    if not args.overfit:
        args.rec_dir=''
    print(args)
    return args
