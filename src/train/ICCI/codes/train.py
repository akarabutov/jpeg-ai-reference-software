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

import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
import numpy as np
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    #### distributed training settings
    opt['dist'] = False
    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                        and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                        screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        version = float(torch.__version__[0:3])
        if version >= 1.1:  # PyTorch 1.1
            from torch.utils.tensorboard import SummaryWriter
        else:
            logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                pbar = util.ProgressBar(len(val_loader))
                if val_set.opt['YUV'] == 420 or val_set.opt['YUV'] == 444:
                    avg_psnr_Y = 0.
                    avg_psnr_U = 0.
                    avg_psnr_V = 0.
                else:
                    avg_psnr = 0.
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    if val_set.opt['YUV']==420 or val_set.opt['YUV']==444:
                        if '10bit' in img_name:
                            sr_img = util.tensor2img(visuals['rlt'], out_type=np.uint16)  # uint16
                            gt_img = util.tensor2img(visuals['GT'], out_type=np.uint16)  # uint8
                        else:
                            sr_img = util.tensor2img(visuals['rlt'])
                            gt_img = util.tensor2img(visuals['GT'])
                        util.save_img(sr_img, save_img_path, mode='YUV')
                    else:
                        sr_img = util.tensor2img(visuals['rlt'], format = 'nYUV')  # uint8
                        gt_img = util.tensor2img(visuals['GT'], format = 'nYUV')  # uint8
                        util.save_img(sr_img, save_img_path, mode='RGB')

                    # calculate PSNR
                    #sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                    if val_set.opt['YUV'] == 420:
                        temp_H = sr_img.shape[0]
                        temp_W = int(sr_img.shape[1]/1.5)
                        temp_uvH = int(0.5*temp_H)
                        temp_uvW = int(0.5*temp_W)
                        avg_psnr_Y += util.calculate_psnr(sr_img[0:temp_H,0:temp_W], gt_img[0:temp_H,0:temp_W])
                        avg_psnr_U += util.calculate_psnr(sr_img[0:temp_uvH, temp_W:temp_W+temp_uvW], gt_img[0:temp_uvH, temp_W:temp_W+temp_uvW])
                        avg_psnr_V += util.calculate_psnr(sr_img[temp_uvH:temp_H,temp_W:temp_W+temp_uvW], gt_img[temp_uvH:temp_H,temp_W:temp_W+temp_uvW])
                    elif val_set.opt['YUV'] == 444:
                        avg_psnr_Y += util.calculate_psnr(sr_img[:,:,0], gt_img[:,:,0])
                        avg_psnr_U += util.calculate_psnr(sr_img[:,:,1], gt_img[:,:,1])
                        avg_psnr_V += util.calculate_psnr(sr_img[:,:,2], gt_img[:,:,2])
                    else:
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                    pbar.update('Test {}'.format(img_name))
                if val_set.opt['YUV'] == 420 or val_set.opt['YUV'] == 444:
                    avg_psnr_Y = avg_psnr_Y / idx
                    avg_psnr_U = avg_psnr_U / idx
                    avg_psnr_V = avg_psnr_V / idx
                    # log
                    logger.info('# Validation # PSNR_Y: {:.4e}, PSNR_U: {:.4e}, PSNR_V: {:.4e}'.format(avg_psnr_Y,avg_psnr_U,avg_psnr_V))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnrY', avg_psnr_Y, current_step)
                        tb_logger.add_scalar('psnrU', avg_psnr_U, current_step)
                        tb_logger.add_scalar('psnrV', avg_psnr_V, current_step)
                else:
                    avg_psnr = avg_psnr / idx
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar('psnr', avg_psnr, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    tb_logger.close()


if __name__ == '__main__':
    main()
