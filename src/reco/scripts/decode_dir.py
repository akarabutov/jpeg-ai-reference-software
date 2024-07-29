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
import subprocess
from concurrent.futures import ProcessPoolExecutor as Pool
from datetime import datetime
from multiprocessing import current_process


def logging_str(s, f):
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    print(s)
    f.write(s + '\n')


def start_and_log_str(cmd, f):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        line = p.stdout.readline()
        if not line:
            break
        else:
            logging_str(line.decode('utf-8').replace('\n', ''), f)

    try:
        p.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        return -1

    return p.returncode


def compute(args):
    import sys
    python_path = sys.executable

    gpu_ids = args['gpu_list']
    bitstream_path = args['bitstream_path']
    rec_dir = args['rec_dir']

    if 'MainProcess' in current_process().name:
        gpu_id = gpu_ids[0]
    else:
        gpu_id = gpu_ids[int(current_process().name.split('-')[1]) - 1]
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f'Start processing of {bitstream_path}')

    log_path_dec = None
    filename = bitstream_path.split('/')[-1]

    log_path_dec = rec_dir + '/' + filename + '_log.txt'
    os.makedirs(os.path.dirname(log_path_dec), exist_ok=True)
    with open(log_path_dec, 'w') as f:
        cmd = [
            python_path, '-m', 'src.reco.coders.decoder', bitstream_path, '-r', rec_dir,
            '--device', 'gpu' if gpu_id >= 0 else 'cpu'
        ]
        logging_str(
            r'Start decoder on ' + (f'GPU {gpu_id}' if gpu_id >= 0 else 'CPU') +
            f": {' '.join(cmd)}\n", f)
        dec_return_code = start_and_log_str(cmd, f)

        if dec_return_code != 0:
            logging_str(
                'Decoder was terminated with error, return code {}'.format(dec_return_code), f)


def convert_str2list(classes):
    classes_list = []
    cum_str = ''
    for c in classes:
        if c.isalpha() and len(cum_str) > 0:
            classes_list.append(cum_str)
            cum_str = ''
        cum_str = cum_str + c
    if len(cum_str) > 0:
        classes_list.append(cum_str)
    return classes_list


def main(input_path, output_path, device, gpu_list):

    os.makedirs(output_path, exist_ok=True)

    num_threads = len(gpu_list)

    if device == 'cpu':
        print(f'Start simulations on {len(gpu_list)} CPUs')
    else:
        print(f"List of GPUs to be used: {', '.join(str(x) for x in gpu_list)}")

    configurations = []

    for root, _, files in os.walk(input_path):
        for filename in files:
            if not filename.lower().endswith('.bits'):
                continue
            bitstream_path = os.path.join(root, filename)
            configurations.append({
                'bitstream_path': bitstream_path,
                'rec_dir': output_path,
                'gpu_list': gpu_list
            })
    if num_threads == 1:
        for configuration in configurations:
            compute(configuration)
    else:
        with Pool(num_threads) as p:
            p.map(compute, configurations)


def def_base_parser():
    from src.codec import get_codec_name, get_codec_version

    prog = f'{get_codec_name()} Decoder [{get_codec_version()}]'
    this = argparse.ArgumentParser(prog=prog)

    this.add_argument('input_path', type=str, default='', help='Path to ImageNet Val dataset')
    this.add_argument('output_path', type=str, default='', help='Path to output')
    this.add_argument('--device', help='Device', default='gpu', choices=['cpu', 'gpu'])

    this.add_argument('-gpu_greedy',
                      dest='gpu_greedy',
                      action='store_true',
                      help='use gpus even if they are already in use')
    this.add_argument('-gpu_max', type=int, help='maximum number of gpus tu use', default=None)
    this.add_argument('-gpu_ids', type=str, default=None, help='maximum number of gpus tu use')
    this.add_argument('--cpu_threads_limit',
                      default=-1,
                      type=int,
                      help=r'Maximum number of used threads '
                      r'(works only with device equal to "cpu")')

    return this


def device_list(cur_args):
    if cur_args['device'] == 'cpu':
        import multiprocessing
        num = multiprocessing.cpu_count()
        if cur_args['cpu_threads_limit'] != -1:
            num = min(num, cur_args['cpu_threads_limit'])
        gpu_ena_list = [-x - 1 for x in range(num)]
    else:
        import GPUtil

        # Get list with available GPUs
        gpu_ids = GPUtil.getAvailable(limit=float('inf'))
        if cur_args['gpu_greedy']:
            gpu_ids = [gpu.id for gpu in GPUtil.getGPUs()]
        if cur_args['gpu_max'] is not None:
            gpu_ids = gpu_ids[:cur_args['gpu_max']]
        if cur_args['gpu_ids'] is not None:
            gpu_ids = [int(s) for s in cur_args['gpu_ids'].split(',')]

        # List of enabled GPU
        gpu_ena_str = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if len(gpu_ena_str) == 0:
            gpu_ena_list = gpu_ids
        else:
            gpu_ena_list = [
                int(x.strip()) for x in gpu_ena_str.split(',')
                if len(x.strip()) > 0 and int(x.strip()) in gpu_ids
            ]

    return gpu_ena_list


if __name__ == '__main__':

    # Import default config file name
    ap = def_base_parser()

    cur_args = ap.parse_args()
    cur_args = vars(cur_args)
    print(ap.prog)

    gpu_ena_list = device_list(cur_args)
    del cur_args['gpu_max']
    del cur_args['gpu_greedy']
    del cur_args['gpu_ids']
    del cur_args['cpu_threads_limit']

    start_time = datetime.now()
    print()
    print('START:', start_time.time())
    print()

    main(**cur_args, gpu_list=gpu_ena_list)
    finish_time = datetime.now()
    print()
    print('FINISH:', finish_time.time())
    print('TOTAL:', finish_time - start_time)
    print()
