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
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import torch
import commentjson
from collections import OrderedDict
from contextlib import closing
from functools import partial
from multiprocessing import Pool

from scripts.acc_train_scripts.report_bdrate_results import BDRateReporter
from scripts.acc_train_scripts.smart_copy_tree import \
    smart_copy_tree  # noqa: E402
from scripts.acc_train_scripts.utils import (copy_model_for_test, run_test, version_check)
from src.train.CCS.acc_train.utils import get_args
from scripts.split_cp import split_cp
from src.codec.utils.templite import Templite

def load_train_configuration(file_path, args, **kwargs):
    t = Templite(filename=os.path.abspath(file_path))
    train_cfg = commentjson.loads(t.render(args=args, **kwargs))
    return train_cfg


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def search_end(file_name):
    is_end = False
    with open(file_name, 'r') as f:
        all_lines = f.readlines()
        right = len(all_lines)
        left = max(0, right - 5)
        for id_line in range(left, right):
            if 'end of train' in all_lines[id_line]:
                is_end = True
    return is_end


def run_stages_for_one_beta(beta, args, beta_2_gpus, train_cfg):

    mse_weight = args.mse_weight 
    batch_size = args.batch_size

    # TODO: move to config json file, dqyu
    file_path = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))

    if args.lst == '':
        beta_2_training_list = {float(k) : v for k,v in train_cfg['beta_2_training_list'].items()}
        args.lst = beta_2_training_list[beta]

    common_parameters = [
        '--data_dir',
        f'{args.data_dir}',
        '--lst',
        f'{args.lst}',
        '--val_data_dir',
        f'{args.val_data_dir}',
        '--val_lst',
        f'{args.val_lst}',
        '--batch_size',
        f'{batch_size}',
        '--seed',
        f'{args.seed}',
        '--test_data_dir',
        f'{args.test_data_dir}',
        '--N',
        f'{args.N}',
        '--N_UV',
        f'{args.N_UV}',
        '--hyper_decoder_type',
        f'{args.hyper_decoder_type}',
        '--hyper_scale_decoder_type',
        f'{args.hyper_scale_decoder_type}',
        '--mse_weight',
        f'{mse_weight}',
        '--beta',
        f'{beta}',  # not used by training script, but passed on to inference script (automatic testing)
        '--use_automatic_testing',
        f'{args.use_automatic_testing}',
        '--automatic_testing_epoch_period',
        f'{args.automatic_testing_epoch_period}',
        '--enable_gvae',
        f'{args.enable_gvae}',
        '--amp',
        f'{args.amp}',
        '--sigma_quant_level',
        f'{args.sigma_quant_level}',
        '--sigma_quant_max',
        f'{args.sigma_quant_max}',
        '--sigma_quant_min',
        f'{args.sigma_quant_min}',
        '--cube_flag_thre',
        f'{args.cube_flag_thre if beta < 0.5 else 0.0}',  # set cube_flag_thre=0.0 for highest rate
        '--loss_weights',
        f'{args.loss_weights}',
        f'--l1',
        f'{args.l1}',
        '--opt_type',
        f'{args.opt_type}',
        '--skip_thre',
        f'{args.skip_thre}',
        '--rec_dir',
        f'{args.rec_dir}',
        '--cfg_path'
    ]
    common_parameters += args.cfg_path
    common_parameters += ['--vae_encoder_type_list'] + args.vae_encoder_type_list
    common_parameters += ['--vae_decoder_type_list'] + args.vae_decoder_type_list
    if len(args.frozen_part) > 0:
        common_parameters += "--frozen_part" + args.frozen_part
    
    if args.overfit:
        common_parameters.append('--overfit')

    beta_2_betaList = {float(k) : v for k,v in train_cfg['beta_2_betaList'].items()}
    beta_list_stageII = str(beta)
    beta_list_stageIII_IV = beta_2_betaList[beta]

    beta_2_msssim_weight = {float(k) : v for k,v in train_cfg['beta_2_msssim_weight'].items()}
    msssim_weight = beta_2_msssim_weight[beta]

    beta_2_loss_factors = {float(k) : v for k,v in train_cfg['beta_2_loss_factors'].items()}
    loss_factors = beta_2_loss_factors[beta]

    # script will automatica iterate over stages and lauch them.
    # in order they are in the OrderedDict
    # best model of previous stage in OrderedDict will be used to
    # start the next stage
    train_stages = load_train_configuration(args.train_stages_json, args, beta=beta, beta_list_stageII=beta_list_stageII, beta_list_stageIII_IV=beta_list_stageIII_IV, msssim_weight=msssim_weight, loss_factors=loss_factors)

    assigned_gpus = beta_2_gpus[beta]
    num_gpus = len(assigned_gpus.split(','))
    stages_lst = train_cfg['stages']

    previous_stage = args.resume_from_stage
    for stage in stages_lst:
        config = train_stages[stage]
        port = find_free_port()
        if stage == "Data_Collection":
            assigned_gpus = str(beta_2_gpus[beta].split(',')[0])
            num_gpus = len(assigned_gpus.split(','))

        # DDP setup
        distributed_launch_cmd = [
            sys.executable,
            '-m',
            'torch.distributed.launch',
            f'--nproc_per_node={num_gpus}',
            '--master_addr=127.0.0.1',
            f'--master_port={port}',
            '-m',
            'src.train.CCS.acc_train.multistages_train.train',
        ]

        # add parameters for current stage
        train_cmd = distributed_launch_cmd + common_parameters + config

        # set output path for current beta's models
        epoch_out_dir = f'{args.train_url}/{stage}/{beta}'
        train_cmd += ['--train_url', epoch_out_dir]

        # add resume from last stage best
        if previous_stage:
            resume_arg = ['--resume', f'{args.train_url}/{previous_stage}/{beta}/best.pth']
        else:
            resume_arg = []
        previous_stage = stage

        # launch taining
        train_cmd = [str(element) for element in train_cmd]  # everything needs to be string
        current_env = os.environ.copy()
        current_env['CUDA_VISIBLE_DEVICES'] = assigned_gpus
        current_env['OMP_NUM_THREADS'] = '1'

        os.makedirs(f'{args.train_url}/log/{stage}', exist_ok=True)
        with open(f'{args.train_url}/log/{stage}/beta{beta}_stdout.log', 'w') as stdout_log, \
             open(f'{args.train_url}/log/{stage}/beta{beta}_stderr.log', 'w') as stderr_log:
            while True:
                # print('Environment:')
                # print(current_env)
                print(f'Launching train for {stage}, beta {beta}:')
                if resume_arg:
                    print(f'Resuming from {resume_arg[1]}')
                print(' '.join(train_cmd))

                proc = subprocess.Popen(train_cmd + resume_arg,
                                        env=current_env,
                                        stdout=stdout_log,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)

                # also show stderr in parents log
                for line in proc.stderr:
                    print(line, end='')
                    stderr_log.write(line)
                proc.wait()

                #if (proc.returncode == 0) or (search_end(stdout_log.name)):
                if (search_end(stdout_log.name)):
                    break
                elif not args.automatic_resume_on_crash:
                    raise subprocess.CalledProcessError(proc.returncode, proc.args, None,
                                                        proc.stderr)
                else:
                    print(f'Failed train for {stage}, beta {beta}. Restarting')
                    if os.path.isdir(epoch_out_dir):
                        epoch_models = os.listdir(epoch_out_dir)
                        # remove tests and best from listing
                        epoch_models = [file for file in epoch_models if 'test' not in file]
                        epoch_models = [file for file in epoch_models if 'best' not in file]
                        finished_epochs = [int(file.split('.pth')[0]) for file in epoch_models]
                        if finished_epochs:
                            last_epoch = sorted(finished_epochs)[-1]
                            resume_arg = [
                                '--resume', f'{args.train_url}/{stage}/{beta}/{last_epoch}.pth',
                                '--resume_opt', '1'
                            ]
                        else:
                            pass
                            # none finisehd so far.
                            # nothing to do here. resume_arg will already be [] for first stage or set accordingly for other stages


def merge_automatic_test_results(results_dir, train_cfg):

    # root_dir = Path( results_dir)

    #betas = [
    #    0.002,
    #    0.012,
    #    0.075,
    #    0.5,
    #]

    #stages = [
    #    'MSE_FixedRate_64', 'Mixed_FixedRate_36', 'Mixed_FixedRate_OnlyDec_20',
    #    'MSE_VariableRate_12'
    #]
    stages = train_cfg['stages']
    betas = list(train_cfg['beta_2_gpus'].keys())

    for stage in stages:
        # take first beta, check how many epocs we already have tests for
        dir_first_beta = os.path.join(results_dir, f'{stage}', f'{betas[0]}')
        if not os.path.isdir(dir_first_beta):
            # this stage is not ready yet
            continue
        content_names = [
            entry for entry in os.listdir(dir_first_beta)
            if os.path.isdir(os.path.join(dir_first_beta, entry))
        ]

        available_epochs = [
            int(dir_name.split('_test')[0]) for dir_name in content_names if '_test' in dir_name
        ]
        available_epochs = sorted(available_epochs)

        for epoch in available_epochs:
            collected_results_dir = os.path.join(results_dir, f'{stage}', 'results',
                                                 f'epoch{epoch}')
            os.makedirs(collected_results_dir, exist_ok=True)

            # check if we have results for all betas
            have_resuls_for_all_betas = True
            for beta in betas:
                beta_result_dir = os.path.join(results_dir, f'{stage}', f'{beta}', f'{epoch}_test',
                                               f'{beta}')
                if not os.path.isdir(beta_result_dir):
                    have_resuls_for_all_betas = False
            if not have_resuls_for_all_betas:
                continue

            # we have results for all betas
            for beta in betas:
                beta_result_dir = os.path.join(results_dir, f'{stage}', f'{beta}', f'{epoch}_test',
                                               f'{beta}')
                smart_copy_tree(beta_result_dir, collected_results_dir, dirs_exist_ok=True)

            cmd = [sys.executable, 'collect_results.py']
            script_path = os.path.join(collected_results_dir, 'collect_results.py')
            if os.path.isfile(script_path):
                try:
                    subprocess.run(cmd,
                                   cwd=collected_results_dir,
                                   check=True,
                                   stdout=subprocess.DEVNULL)
                except subprocess.CalledProcessError as exc:
                    traceback.print_exc(exc)


def get_all_gpus(beta_2_gpus: dict) -> str:
    available_gpu_ids = []
    for _, value in beta_2_gpus.items():
        for id in value.split(','):
            available_gpu_ids.append(id)
    # merge ids
    available_gpu_ids = set(available_gpu_ids)
    return ','.join(available_gpu_ids)


def run_test_for_stages_best_models(args, beta_2_gpus, train_output_dir, train_cfg):
    file_path = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))

    print("Doing tests of each each stages 'best' models.")
    available_gpu_ids = get_all_gpus(beta_2_gpus)
    #stages = [
    #    'MSE_FixedRate_64', 'Mixed_FixedRate_36', 'Mixed_FixedRate_OnlyDec_20',
    #    'MSE_VariableRate_12'
    #]
    #betas = [
    #    0.002,
    #    0.012,
    #    0.075,
    #    0.5,
    #]
    stages = train_cfg['stages']
    betas = list(train_cfg['beta_2_gpus'].keys())
    epoch = 'best'

    for stage in stages:
        with tempfile.TemporaryDirectory() as models_dir_path: 
            cfgs = list()
            for cfg in args.cfg_path:
                cfgs.append(os.path.join(proj_dir, os.path.join('cfg/', cfg)))
            vae_lists = args.vae_decoder_type_list + args.vae_encoder_type_list
            vae_list_uniq = list(set(vae_lists))
            for beta in betas:
                pth_files = copy_model_for_test(
                    os.path.join(models_dir_path, "VM_tmp"),
                    beta,
                    epoch,
                    os.path.join(train_output_dir, f'{stage}/{beta}'),
                )
                for pfn in pth_files:
                    data = torch.load(pfn)
                    split_cp(data, os.path.basename(pfn), vae_list_uniq, models_dir_path)

            current_env = os.environ.copy()
            current_env['CUDA_VISIBLE_DEVICES'] = available_gpu_ids
            current_env['OMP_NUM_THREADS'] = '1'
            test_results_dir = os.path.join(train_output_dir, f'{stage}', 'results', f'epoch{epoch}')

            run_test(test_results_dir, epoch, args.test_data_dir, models_dir_path, cfgs, current_env, args)


def main(args=None):
    version_check()
    if not args:
        args = get_args()  # in cloud train. args will be given by wrapping script

    # work_dir = proj_dir
    # os.chdir(work_dir)
    with open(args.train_cfg_json, "r") as f:
        train_cfg = commentjson.load(f)

    # copy test data to local dir
    with tempfile.TemporaryDirectory() as local_test_data_dir:
        smart_copy_tree(args.test_data_dir, f'{local_test_data_dir}/tmp')
        args.test_data_dir = f'{local_test_data_dir}/tmp'

        # TODO: query number of gpus and assigne load smartly
        train_cfg['beta_2_gpus'] = { float(k): v for k,v in train_cfg['beta_2_gpus'].items() }
        beta_2_gpus = train_cfg['beta_2_gpus']

        betas = beta_2_gpus.keys()

        train_url = args.train_url
        os.makedirs(f'{args.train_url}', exist_ok=True)

        bdrate_reporter = BDRateReporter(train_url)
        num_workers = 4
        with tempfile.TemporaryDirectory() as train_output_dir:
            # copy files from previous training dir
            if args.copy_to_train_url_dir:
                smart_copy_tree(args.copy_to_train_url_dir, train_output_dir, dirs_exist_ok=True)
            
            args.train_url = train_output_dir
            with Pool(num_workers) as workers:

                results = workers.map_async(
                    partial(run_stages_for_one_beta, args=args, beta_2_gpus=beta_2_gpus, train_cfg=train_cfg), betas)

                # Wait for every task to finish
                while not results.ready():
                    time.sleep(60 * 0.5)  # every 30s

                    # periodically copy output
                    smart_copy_tree(train_output_dir, train_url, dirs_exist_ok=True)

                    if args.use_automatic_testing and args.generate_test_summary:
                        merge_automatic_test_results(train_url, train_cfg=train_cfg)
                        bdrate_reporter.report_bdrate_results()

                    print('Still waiting')
            smart_copy_tree(train_output_dir, train_url, dirs_exist_ok=True)
            workers.close()
            workers.join()

            if args.use_automatic_testing and args.use_automatic_testing_best:
                run_test_for_stages_best_models(args, beta_2_gpus, train_output_dir, train_cfg=train_cfg)
                smart_copy_tree(train_output_dir, train_url, dirs_exist_ok=True)

            # final copy of output
            smart_copy_tree(train_output_dir, train_url, dirs_exist_ok=True)
        if args.use_automatic_testing and args.generate_test_summary:
            merge_automatic_test_results(train_url, train_cfg=train_cfg)
            bdrate_reporter.report_bdrate_results()
            bdrate_reporter.close_summary_writer()


if __name__ == '__main__':
    main()
