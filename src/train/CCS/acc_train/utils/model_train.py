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

import gc
import time
import warnings
import tempfile
import json

import numpy as np
import torch
import torch.distributed as dist

try:
    from torch.cuda.amp import GradScaler
except:  # noqa: E722
    raise ImportError('AMP is needed.')
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from prettytable import PrettyTable
from torch.nn.utils import clip_grad_norm_ as cgn
from torch.utils.tensorboard import SummaryWriter

from scripts.acc_train_scripts.utils import copy_model_for_test, run_test
from scripts.split_cp import split_cp
from src.train.CCS.acc_train.utils.mp_env import set_random
from src.train.CCS.utils import AverageMeter

warnings.filterwarnings('ignore')

__all__ = ['train']


def has_nan(tensor_list):
    ret = False
    for tensor in tensor_list:
        if torch.isnan(tensor).item():
            ret = True
    return ret


def resume_epoch(scheduler, optimizer, train_loader, args):
    if args.resume_opt:
        ckpt_path = args.resume
        if os.path.isfile(args.resume):
            print("=> loading optimizer from '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']

            len_data_loader = len(train_loader)
            for epoch in range(args.start_epoch):
                if train_loader.sampler is not None:
                    train_loader.sampler.set_epoch(epoch)
                if args.lr_type == 'step.epoch':
                    scheduler.step()
                elif args.lr_type == 'reduce_on_plateau.epoch':
                    scheduler.step(best_loss)
                for _ in range(len_data_loader):
                    # start_time = time.time()
                    if args.lr_type.endswith('step'):
                        scheduler.step()
            return best_loss
        else:
            raise ValueError(f'{args.resume} is not file')
    else:
        return 10e9


def init_summary_writer(tensor_board_out_dir: Path) -> SummaryWriter:
    summary_writer = SummaryWriter(tensor_board_out_dir)

    # add a plot showing train and validation loss together
    layout = {
        'Train vs. validation loss': {
            'loss': ['Multiline', ['rd_loss/train', 'reg_loss/train', 'Loss/validation']],
        },
    }
    summary_writer.add_custom_scalars(layout)

    return summary_writer


def update_tensorboard(tb_summary_writer: SummaryWriter, epoch: int, train_meters,
                       val_meters) -> None:

    for key, meter in train_meters.items():
        if key == 'Time':
            continue
        tb_summary_writer.add_scalar(f'{key}/train', meter.avg, epoch)

    for key, meter in val_meters.items():
        if key == 'Time':
            continue
        tb_summary_writer.add_scalar(f'{key}/validation', meter.avg, epoch)


def train(train_loader, val_loader, net, criterion, optimizer, scheduler, args):
    if not os.path.exists(args.train_url):
        os.makedirs(args.train_url, exist_ok=True)

    stage = Path(args.train_url).parent.name
    tensor_board_out_dir = Path(args.train_url).parents[1] / f'tensorboard/{stage}/{args.beta}'
    tensor_board_out_dir.mkdir(exist_ok=True, parents=True)

    tb_summary_writer = init_summary_writer(tensor_board_out_dir)

    time_all_epochs = timedelta()
    time_all_tests = timedelta()

    is_best = False
    best_loss = resume_epoch(scheduler, optimizer, train_loader, args)

    if dist.get_rank() == 0 and args.collect_only:
        data_collect_time0 = time.time()
        data_collection(train_loader, net, criterion, args)
        net.save_model_and_opt(args, args.start_epoch, optimizer, best_loss, model_prefix=str(args.start_epoch), is_best=True)
        data_collect_time1 = time.time()
        print(f'Time of data collection is {data_collect_time1 - data_collect_time0} seconds')
        return

    if not args.resume_opt:
        val_meters = collect_meters(validation(val_loader, net, criterion, args, -1))
        val_result_printer = ResultPrinter(epoch_str='Keys', headers=list(val_meters.keys()))
        val_result_printer.insert_row(val_meters, 'Before train', 'avg')
        if dist.get_rank() == 0:
            val_result_printer.print()
            output_fn = os.path.join(args.train_url, "val_results.json")
            val_result_printer.store(output_fn)
            print(f'Stored validation results to a file {output_fn}')
    else:
        val_result_printer = None

    # scaler = GradScaler() if args.amp else None
    count_trained_epochs = 0
    for epoch in range(args.start_epoch, args.epochs):
        if count_trained_epochs == 100 and not args.overfit:
            exit()
        count_trained_epochs += 1
        start_time_epoch = datetime.now()
        set_random(args.seed + epoch)
        if args.lr_type == 'step.epoch':
            scheduler.step()
        elif args.lr_type == 'reduce_on_plateau.epoch':
            scheduler.step(best_loss)
        train_meters = train_one_epoch(train_loader, net, criterion, optimizer, scheduler, epoch,
                                       args)
        val_meters = collect_meters(validation(val_loader, net, criterion, args, epoch))
        if val_result_printer is None:
            val_result_printer = ResultPrinter(epoch_str='Keys', headers=list(val_meters.keys()))
        val_result_printer.insert_row(val_meters, f'Epoch:[{epoch}]', 'avg')
        is_best = (val_meters['Loss'].avg < best_loss)
        if is_best:
            best_loss = val_meters['Loss'].avg
            val_result_printer.insert_row(val_meters, 'best', 'avg')
            print(f'Best so far is epoch {epoch}')

        torch.distributed.barrier()
        if dist.get_rank() == 0:
            val_result_printer.print()
            output_fn = os.path.join(args.train_url, "val_results.json")
            val_result_printer.store(output_fn)
            
            update_tensorboard(tb_summary_writer, epoch, train_meters, val_meters)

            # delete old data in temporary direcotry created by parent process, it should already have been copied.
            # avoiding out of space of temporary direcotry
            # best epoch should be kept. last epoch should be kept, if needed for resume from crash, it is saved directly after this
            output_paths = os.listdir(args.train_url)
            for path in output_paths:
                abs_path = os.path.join(args.train_url, path)
                if os.path.isdir(abs_path):
                    shutil.rmtree(abs_path)
                else:
                    if path in ['best.pth', 'val_results.json']:
                        continue
                    else:
                        os.remove(abs_path)

            net.save_model_and_opt(args,
                                   epoch,
                                   optimizer,
                                   best_loss,
                                   model_prefix=str(epoch),
                                   is_best=is_best)

            finish_time_epoch = datetime.now()
            time_for_this_epoch = datetime.now() - start_time_epoch
            print(f'Epoch {epoch} took {time_for_this_epoch}')
            time_all_epochs += time_for_this_epoch
            start_time_test = finish_time_epoch

            # test every n epochs
            if args.use_automatic_testing:
                proj_dir = Path(__file__).parents[5]

                test_after_num_epochs = args.automatic_testing_epoch_period

                if (epoch % test_after_num_epochs == 0  # every few epochs
                        or args.epochs - 1 == epoch):  # always include last epoch of stage

                    # determine trainig type
                    with tempfile.TemporaryDirectory() as models_dir_path: 
                        print(f'Doing inference test for epoch {epoch}')
                        cfgs = [os.path.join(proj_dir, os.path.join('cfg/', x)) for x in args.cfg_path]
                        model_path_tmp = os.path.join(models_dir_path, 'VM_tmp')

                        copy_model_for_test(model_path_tmp,
                                            args.beta,
                                            epoch,
                                            args.train_url,
                                            create_missing_models=True)
                        op_list = list(set(args.vae_encoder_type_list + args.vae_decoder_type_list))
                        for fn in os.listdir(model_path_tmp):
                            data = torch.load(os.path.join(model_path_tmp, fn))
                            split_cp(data, fn, op_list, models_dir_path)

                        test_results_dir = os.path.join(args.train_url, f'{epoch}_test',
                                                        f'{args.beta}')

                        run_test(test_results_dir, epoch, args.test_data_dir, models_dir_path, cfgs, os.environ.copy(), args, args.beta)

                time_for_this_test = datetime.now() - start_time_test
                print(f'Test {epoch} took {time_for_this_test}')
                time_all_tests += time_for_this_test
        torch.distributed.barrier()

    tb_summary_writer.close()
    print(f'Total time (train and test): {time_all_epochs + time_all_tests}')
    print(f'Total time train: {time_all_epochs}')
    print(f'Total time test: {time_all_tests}')


class ResultPrinter:
    default_headers = (
        'Time',
        'Data',
        'Rate',
        'Hyper_rate',
        'Y_PSNR',
        'U_PSNR',
        'V_PSNR',
        'Loss',
    )

    def __init__(self, epoch_str='', headers=None):
        if headers is None:
            headers = list(self.default_headers)  # copy
        self._headers = headers
        self._table = PrettyTable()
        self._table.field_names = [epoch_str] + self._headers

    def insert_row(self, meters, call_str='', meter_attribute='val'):
        if meter_attribute == 'val':
            results = {meter.name: meter.val for meter in meters.values()}
        elif meter_attribute == 'avg':
            results = {meter.name: meter.avg for meter in meters.values()}
        else:
            assert False, 'Not Implemented'

        self.update(results, call_str)

    def update(self, results, call_str=''):
        row = [call_str] + [f'{results[header]:.4f}' for header in self._headers]
        self._table.add_row(row)

    def print(self):
        print(self._table)
        
    @staticmethod
    def get_dict_from_row(field_names, row):
        ans = dict()
        for k,v in zip(field_names, row):
            ans[k] = v
        return ans
        
    def get_dict(self):
        """Generate dictionary based on a table
        """
        ans = list()
        f = self._table.field_names
        for row in self._table._rows:
            ans.append(self.get_dict_from_row(f, row))
        return ans
        
    def store(self, json_path: str):
        """Store the table to a json file

        Args:
            json_path (str): path to an output json file
        """
        output_arr = self.get_dict()
        with open(json_path, 'w') as f:
            json.dump(output_arr, f)
        
        


def update_meters(batch_size, meters, kwargs):
    for key, val in kwargs.items():
        if key != 'rec_data':
            if key != 'Time':
                meters[key].update(val, batch_size)
            else:
                meters[key].update(val)


def create_meters(
        keys=['Time', 'Data', 'Rate', 'Hyper_rate', 'Y_PSNR', 'U_PSNR', 'V_PSNR', 'Loss']):
    meters = [AverageMeter(key) for key in keys]
    meters_dict = {meter.name: meter for meter in meters}
    return meters_dict


def print_meters(meters_dict, keys=['Time'], prefix=''):
    row = prefix
    for key in keys:
        if key != 'Time':
            row += f'{key} {meters_dict[key].val:.4e} '
        else:
            row += f'{key} {meters_dict[key].val:.4f}({meters_dict[key].avg:.4f}) '
    print(row)


def merge_meters(meters_list):
    ret_meters = meters_list[0]
    for meters in meters_list[1:]:
        for meter_name, meter in meters:
            ret_meters[meter_name].update(meter.avg, n=meter.count)
    return ret_meters


def collect_meters(meters):
    group = dist.group.WORLD
    ret_meters = create_meters(meters.keys())
    device = torch.device(dist.get_rank() if torch.cuda.is_available() else 'cpu')
    tensor_list = [torch.zeros(2).to(device) for _ in range(dist.get_world_size())]
    for key, meter in meters.items():
        dist.all_gather(tensor_list,
                        torch.tensor([meter.avg, meter.count],
                                     dtype=torch.float32).to(device),
                        group=group,
                        async_op=False)
        # NCCL do not support param_net
        # if dist.get_rank() == root:
        #    dist.param_net(torch.tensor([meter.avg, meter.count]), param_net_list=tensor_list, group=group)
        # else:
        #    dist.param_net(torch.tensor([meter.avg, meter.count]), dst=root, group=group)
        for tensor in tensor_list:
            avg, count = tensor[0].item(), tensor[1].item()
            ret_meters[key].update(avg, count)
    return ret_meters


def grad_has_nan(param_groups):
    device = param_groups[0]['params'][0].device
    is_nan = torch.tensor(False, device=device, dtype=torch.bool)
    world_wide_nan = [
        torch.tensor(False, device=device, dtype=torch.bool) for _ in range(dist.get_world_size())
    ]
    for group in param_groups:
        if torch.any(is_nan):
            break
        for p in group['params']:
            if p.grad is None:
                continue
            if torch.isnan(p.grad).any():
                is_nan = torch.tensor(True, device=device, dtype=torch.bool)
                break
    torch.distributed.barrier()
    dist.all_gather(world_wide_nan, is_nan, group=None, async_op=False)
    ret = False
    for item in world_wide_nan:
        if torch.any(item):
            ret = True
    return ret


def get_nan_grad(net):
    nan_grads_name = list()
    for (m_name, m) in net.named_model_list:
        for param_name, param in m.named_parameters():
            grad_nan = torch.any(torch.isnan(param.grad)).item()
            if grad_nan:
                full_param_name = m_name + '.' + param_name,
                print(dist.get_rank(), full_param_name[0],
                      np.nanmax(param.grad.detach().cpu().numpy()),
                      np.nanmin(param.grad.detach().cpu().numpy()))
                nan_grads_name.append(full_param_name)
    return nan_grads_name


def has_nan_grad(scaler, optimizer):
    device = optimizer.param_groups[0]['params'][0].device
    is_nan = torch.tensor(False, device=device, dtype=torch.bool)
    world_wide_nan = [
        torch.tensor(False, device=device, dtype=torch.bool) for _ in range(dist.get_world_size())
    ]
    if sum(v.item()
           for v in scaler._per_optimizer_states[id(optimizer)]['found_inf_per_device'].values()):
        is_nan = torch.tensor(True, device=device, dtype=torch.bool)
    dist.all_gather(world_wide_nan, is_nan, group=None, async_op=False)
    torch.distributed.barrier()
    ret = False
    for item in world_wide_nan:
        if torch.any(item):
            ret = True
    return ret


def get_total_loss(args, loss_dict):
    loss = loss_dict['rd_loss'] + loss_dict['reg_loss']
    return loss


def train_one_epoch(train_loader, net, criterion, optimizer, scheduler, epoch, args):
    epoch_start_time = time.time()
    net.set_train_mode()
    world_wide_loss = [torch.tensor([0.0], device=net.device) for _ in range(dist.get_world_size())]
    nan_count, use_amp = 0, args.amp
    if train_loader.sampler is not None:
        train_loader.sampler.set_epoch(epoch)
    train_meters = create_meters(['Time', 'rd_loss', 'reg_loss', 'lr', 'scale'])
    torch.cuda.empty_cache()
    gc.collect()
    scaler = GradScaler() if args.amp else None
    for istep, (_, x) in enumerate(train_loader):
        start_time = time.time()
        if args.lr_type.endswith('step'):
            scheduler.step()
        beta = net.beta_list[np.random.randint(0, len(net.beta_list))]
        loss_dict = net.train_forward_to_loss(x, criterion, beta, args, amp=use_amp)
        loss = get_total_loss(args, loss_dict)
        torch.distributed.barrier()
        dist.all_gather(world_wide_loss, loss, group=None, async_op=False)
        if has_nan(world_wide_loss):
            overflow = True
            if use_amp:
                nan_count += 1
                loss_dict = net.train_forward_to_loss(x, criterion, beta, args, amp=False)
                loss = get_total_loss(args, loss_dict)
                if (nan_count / len(train_loader)) > 0.02:
                    use_amp = False
        else:
            overflow = False
        optimizer.zero_grad()
        if use_amp and (not overflow):  # amp and not nan
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for m in net.model_list:
                cgn(m.parameters(), 10)
            scaler.step(optimizer)
            scale_factor = scaler.get_scale()
            is_nan_grad = has_nan_grad(scaler, optimizer)
            if is_nan_grad:
                nan_count += 1
                if (nan_count / len(train_loader)) > 0.02:
                    use_amp = False
            if scaler.get_scale() < 2048.0:
                scaler.update(2048.0)
            else:
                scaler.update()
            if is_nan_grad:
                optimizer.zero_grad()
                loss_dict = net.train_forward_to_loss(x, criterion, beta, args, amp=False)
                loss = get_total_loss(args, loss_dict)
                loss.backward()
                for m in net.model_list:
                    cgn(m.parameters(), 10)
                optimizer.step()
                scale_factor = 16.0
        else:
            loss.backward()
            for m in net.model_list:
                cgn(m.parameters(), 10)
            optimizer.step()
            scale_factor = 32.0
        end_time = time.time()
        update_meters(
            train_loader.batch_size, train_meters,
            dict(
                {
                    'Time': end_time - start_time,
                    'lr': optimizer.param_groups[0]['lr'],
                    'scale': scale_factor,
                }, **loss_dict))
        if dist.get_rank() == 0 and istep % args.print_freq == 0:
            print_meters(train_meters,
                         keys=['Time', 'lr', 'rd_loss', 'reg_loss', 'scale'],
                         prefix=f'[{epoch:d}/{args.epochs:d}][{istep:d}/{len(train_loader):d}] ')
    epoch_end_time = time.time()
    if dist.get_rank() == 0:
        print(f'Train time of epoch {epoch} is {epoch_end_time - epoch_start_time} seconds')
    return train_meters


def validation(data_loader, net, criterion, args, epoch):
    meter_keys = ['Rate', 'Hyper_rate', 'Y_PSNR', 'U_PSNR', 'V_PSNR', 'MSE', 'MSSSIM', 'Loss']
    val_meters = create_meters(meter_keys)
    net.set_eval_mode()
    net.zero_channel_wise_entropy()
    torch.cuda.empty_cache()
    gc.collect()
    for istep, (_, x) in enumerate(data_loader):
        beta = net.beta_list[istep % len(net.beta_list)]
        val_ret = net.val_forward_to_loss(x, criterion, beta, args, amp=False)
        if len(args.rec_dir) > 0:
            from src.codec.common import Image
            img = Image.create_from_tensors(val_ret['rec_data']['xY'], val_ret['rec_data']['xCb'], val_ret['rec_data']['xCr'], [0, 255], color_space='yuv')
            os.makedirs(args.rec_dir, exist_ok=True)
            img.write_file(os.path.join(args.rec_dir, f"iter_{epoch:06d}.png"))
        if val_ret is not None:
            # batch_size = x.shape[0]
            update_meters(x.shape[0], val_meters, val_ret)
    return val_meters


def idx_to_num(idx, min, max, bin_num):
    intv = (max - min) / bin_num
    num = min + idx * intv
    return num


def get_thres(stat, remv_num):
    stat_thres = dict.fromkeys(stat.keys(), {})
    for layer in stat:
        hist = np.array(stat[layer]['StaList'])
        if hist.shape[0] == 0:
            continue
        c, bins = hist.shape

        stat_thres[layer] = dict.fromkeys(('MaxList', 'MinList'), [])
        stat_thres[layer]['MaxList'] = np.zeros(c)                
        stat_thres[layer]['MinList'] = np.zeros(c)                

        for ch in range(c):
            total_num = np.sum(hist[ch])
            hist_cum = np.cumsum(hist[ch])
            num_upper = total_num - remv_num
            num_lower = remv_num

            upper_idx = np.searchsorted(hist_cum, num_upper, side='left')
            lower_idx = np.searchsorted(hist_cum, num_lower, side='right')
            stat_thres[layer]['MaxList'][ch] = min(idx_to_num(upper_idx + 1, stat[layer]['MinList'][ch], stat[layer]['MaxList'][ch], 1000), stat[layer]['MaxList'][ch])
            stat_thres[layer]['MinList'][ch] = max(idx_to_num(lower_idx, stat[layer]['MinList'][ch], stat[layer]['MaxList'][ch], 1000), stat[layer]['MinList'][ch])
        
        stat_thres[layer]['MaxList'] = stat_thres[layer]['MaxList'].tolist()
        stat_thres[layer]['MinList'] = stat_thres[layer]['MinList'].tolist()

    return stat_thres


def set_clip_thres(m, stat, key_list):
    m.clip_thres = dict()
    for k, v in stat.items():
        if k in key_list:
            m.clip_thres[k] = v 


def data_collection(data_loader, net, criterion, args):
    net.set_eval_mode()
    torch.cuda.empty_cache()
    gc.collect()
    stat_Y = {
        'E1B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E2B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E3B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E4B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E5B':{'MaxList':[],'MinList':[],'StaList':[]},

        'E1H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E2H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E3H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E4H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E5H':{'MaxList':[],'MinList':[],'StaList':[]},

        'HE1':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE2':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE3':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE4':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE5':{'MaxList':[],'MinList':[],'StaList':[]},
    }
    stat_UV = {
        'E1B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E2B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E3B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E4B':{'MaxList':[],'MinList':[],'StaList':[]},
        'E5B':{'MaxList':[],'MinList':[],'StaList':[]},

        'E1H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E2H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E3H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E4H':{'MaxList':[],'MinList':[],'StaList':[]},
        'E5H':{'MaxList':[],'MinList':[],'StaList':[]},

        'HE1':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE2':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE3':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE4':{'MaxList':[],'MinList':[],'StaList':[]},
        'HE5':{'MaxList':[],'MinList':[],'StaList':[]},
    }
    print("enter first pass: collect maximum & minimum values")
    for istep, (_, x) in enumerate(data_loader):
        beta = net.beta_list[istep % len(net.beta_list)]
        stat_Y, stat_UV = net.data_collection_forward(x, criterion, beta, args, stat_Y, stat_UV, first_pass=True, amp=False)
        if dist.get_rank() == 0 and istep % args.print_freq == 0:
            print(f'[{istep}/{len(data_loader)}]')
    
    print("enter second pass: collect freuqency distribution")
    for istep, (_, x) in enumerate(data_loader):
        beta = net.beta_list[istep % len(net.beta_list)]
        stat_Y, stat_UV = net.data_collection_forward(x, criterion, beta, args, stat_Y, stat_UV, first_pass=False, amp=False)
        if dist.get_rank() == 0 and istep % args.print_freq == 0:
            print(f'[{istep}/{len(data_loader)}]')

    print("remove extreme values")
    # remove extreme values
    stat_Y_process = get_thres(stat_Y, 100)
    stat_UV_process = get_thres(stat_UV, 100)

    for name in net.encoder_Y.module.coders.keys():
        if 'bop' in name:
            set_clip_thres(net.encoder_Y.module.coders['bop_prim'], stat_Y_process, ['E1B', 'E2B', 'E3B', 'E4B', 'E5B'])
            set_clip_thres(net.encoder_UV.module.coders['bop_sec'], stat_UV_process, ['E1B', 'E2B', 'E3B', 'E4B', 'E5B'])
        elif 'hop' in name:
            set_clip_thres(net.encoder_Y.module.coders['hop_prim'], stat_Y_process, ['E1H', 'E2H', 'E3H', 'E4H', 'E5H'])
            set_clip_thres(net.encoder_UV.module.coders['hop_sec'], stat_UV_process, ['E1H', 'E2H', 'E3H', 'E4H', 'E5H'])

    set_clip_thres(net.hyper_encoder_Y.module, stat_Y_process, ['HE1', 'HE2', 'HE3', 'HE4', 'HE5'])
    set_clip_thres(net.hyper_encoder_UV.module, stat_UV_process, ['HE1', 'HE2', 'HE3', 'HE4', 'HE5'])

    return
