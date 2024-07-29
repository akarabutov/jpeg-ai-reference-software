import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import re

import torch


def install_packages():
    packages = [
        'attrs',
        'decorator',
        'GPUtil',
        'IQA-pytorch',
        'numpy',
        'opencv-python',
        'pandas',
        'prettytable',
        'psutil',
        'ptflops==0.6.5',
        'psnr_hvsm==0.1.0',
        'pynvml',
        'pyrtools',
        'pytorch-msssim',
        'scikit-image',
        'scipy',
        'einops',
        'openpyxl',
        'pybind11', 
        'tensorboard==2.9.0',
        'wheel==0.34.2',
        'scikit-learn-extra==0.3.0',
        'commentjson==0.9.0'        
        # 'bjontegaard==1.1.0',  #TODO use pip package once it is available on cluster
    ]
    cur_script_dir = os.path.dirname(os.path.abspath(__file__))
    workdir = os.path.abspath(os.path.join(cur_script_dir, os.pardir,os.pardir))
    os.chdir(workdir)
    cmd = f'pip install ' + ' '.join(packages)
    print(f"Execute command: {cmd}")
    os.system(cmd)
    # Get a name of the current conda env.
    conda_env_re = re.search('envs\/(?P<name>\w+)\/bin', sys.executable)
    conda_env = "base" if conda_env_re is None else conda_env_re.group('name')
    #build_lib_path = os.path.join(workdir, 'scripts', 'build_cpp_libs.sh')
    build_lib_path = os.path.join(workdir, 'scripts', 'build_train_libs.sh')
    cmd = f"sh {build_lib_path} {conda_env}"
    print(f"Execute command: {cmd}")
    os.system(cmd)




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


def cloud_train():
    is_cloud = False
    try:
        import moxing as mox
        mox.file.shift('os', 'mox')
        is_cloud = True
    except ImportError:
        print('No moxing module, local train!')
    return is_cloud


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
                '--skip_models_check',
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
