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
import gc
import os
import shutil
import subprocess
import sys
import tempfile

import torch

def get_free_tcp_ports(num_ports):
    import random
    import socket
    ret_ports = set()
    while True:
        port = random.randint(7000, 9999)
        try:
            sock = socket.socket()
            sock.bind(('', port))
            ret_ports.add(port)
            if len(ret_ports) == num_ports:
                return list(ret_ports)
            else:
                continue
        except Exception:
            continue


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def version_check():
    print('--- Versions ---')
    print('torch version:', torch.__version__)
    if torch.cuda.is_available():
        print('cuda version:', torch.version.cuda)
        print('cudnn version:', torch.backends.cudnn.version())
        print('cuda device:', torch.cuda.get_device_name())


def copy_model_for_test(model_path, beta, epoch, train_output_dir, create_missing_models=False):

    ans = list()
    os.makedirs(model_path, exist_ok=True)
    # copy model of current epoch for test
    # model_prefix = str(epoch)
    # checkpoint_name = model_prefix + '.pth'
    checkpoint_name = 'best.pth'
    # abs_checkpoint_name = os.path.join(train_output_dir, f'{stage}/{beta}', checkpoint_name)
    abs_checkpoint_name = os.path.join(train_output_dir, checkpoint_name)

    pth_file_current_beta = os.path.join(model_path, f'{beta}.pth')
    shutil.copyfile(abs_checkpoint_name, pth_file_current_beta)
    ans.append(pth_file_current_beta)
    print(f'copy current beta model {abs_checkpoint_name} to {pth_file_current_beta}')

    if create_missing_models:
        # for others create if don't exist
        pth_files = [
            os.path.join(model_path, '0.002.pth'),
            os.path.join(model_path, '0.012.pth'),
            os.path.join(model_path, '0.075.pth'),
            os.path.join(model_path, '0.5.pth'),
        ]
        for pth_file in pth_files:
            # only one will be used for test of current bpp_idx. but all are expected to exit by SW
            if not os.path.isfile(pth_file):
                shutil.copyfile(abs_checkpoint_name, pth_file)
                print(f'copy {abs_checkpoint_name} to {pth_file}')
                ans.append(pth_file)
    return ans


def run_test(
    test_results_dir,
    epoch,
    test_data_dir,
    models_dir_name,
    cfgs,
    current_env,
    args,
    beta=-1,
):
    # Change CUDA_VISIBLE_DEVICES to a single card
    if 'CUDA_VISIBLE_DEVICES' in current_env:
        cur_cards = current_env['CUDA_VISIBLE_DEVICES']
        cur_cards_list = cur_cards.split(',')
        current_env['CUDA_VISIBLE_DEVICES'] = cur_cards_list[0]
    
    print(f'Doing inference test for epoch {epoch}')
    os.makedirs(test_data_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as test_results_dir_tmp:
        print(f'test_results_dir: {test_results_dir}')
        print(f'test_results_dir_tmp: {test_results_dir_tmp}')

        if beta == -1:
            beta_2_target_bbp = {
                '-1': [-1]  # do all beta test
            }
        else:
            # for mapping beta to correct bpp idcs
            beta_2_target_bbp = {
                '0.002': [12],
                '0.012': [25],
                '0.075': [50],
                '0.5': [75],  # needs two since it is used for test of 050 and 075
            }

        for target_bbp in beta_2_target_bbp[str(beta)]:
            test_cmd = [
                sys.executable,
                '-m',
                'src.reco.scripts.eval',
                '--coding_type',
                'enc',
                '--in_dir',
                test_data_dir,
                '--out_dir',
                test_results_dir_tmp,
                #  "--imgs", "00011_TE_1512x2016_8bit_sRGB.png",
                '-target_bpps',
                f'{target_bbp}',
                '--gpu_greedy',
                '--skip_loading_error',
                '--models_dir_name', models_dir_name,
                '-model.CCS_SGMM.tools_common.model_common.common_modules.ckpt_model_name', 'VM_common',
                '--cfg',
            ]
            test_cmd += cfgs
            test_cmd.append('./cfg/test_after_train.json')
            
            test_cmd += [
                '-model.CCS_SGMM.tools_common.model_common.hyper_scale_decoder_type',
                args.hyper_scale_decoder_type
                ]
            test_cmd += [
                '-model.CCS_SGMM.tools_common.model_common.hyper_decoder_type',
                args.hyper_decoder_type
                ]


            print(f'test_cmd: {test_cmd}')

            torch.cuda.empty_cache()
            gc.collect()

            with open(f'{test_results_dir_tmp}/stdout.log',
                      'w') as stdout_log, open(f'{test_results_dir_tmp}/stderr.log',
                                               'w') as stderr_log:
                subprocess.run(test_cmd,
                               env=current_env,
                               stdout=stdout_log,
                               stderr=stderr_log,
                               universal_newlines=True)

        # remove unnecessary files
        dirs_to_remove = ['ori', 'rec', 'bit']
        for directory in dirs_to_remove:
            dir_path = os.path.join(test_results_dir_tmp, directory)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        # files_to_remove = ['collect_results.py', 'cfg.json']
        # for file in files_to_remove:
        #     file_path = os.path.join(test_results_dir_tmp, file)
        #     if os.path.exists(file_path):
        #         os.remove(file_path)

        shutil.copytree(test_results_dir_tmp, test_results_dir)
