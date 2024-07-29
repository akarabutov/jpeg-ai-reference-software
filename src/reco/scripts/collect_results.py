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
import json
import os
import re


# ######################################################################################################################
#  Parameter methods
# ######################################################################################################################
def parser_args():
    root_dir = os.getcwd()

    this = argparse.ArgumentParser('Tool for collecting results')
    ##
    this.add_argument('--root_dir', default=root_dir)
    this.add_argument('--summary', default=None)
    this.add_argument('--enc_log', default=None, help='Directory to logfiles')
    this.add_argument('--dec_log', default=None, help='Directory to logfiles')
    this.add_argument('--cmp_log', default=None, help='Directory to logfiles')
    this.add_argument('--use_kmacpp', default=1, type=int, help='Write KMac/pxl to logs')
    this.add_argument('--rate_num', default=5, type=int, help='Number of rates')
    this.add_argument('--pre_lines',
                      default=0,
                      type=int,
                      help='Number of lines to fill at the section beginning.')
    args, _ = this.parse_known_args()

    root_dir = args.root_dir
    if args.summary is None:
        args.summary = os.path.join(root_dir, 'summary.txt')
    if args.enc_log is None:
        args.enc_log = os.path.join(root_dir, 'log', 'enc')
    if args.dec_log is None:
        args.dec_log = os.path.join(root_dir, 'log', 'dec')
    if args.cmp_log is None:
        args.cmp_log = os.path.join(root_dir, 'log', 'compare')

    return args


# ######################################################################################################################
#  service methods
# ######################################################################################################################
def check_enc_log_file(enc_log_dir, fname):
    path = os.path.join(enc_log_dir, fname)
    if os.path.isfile(path):
        return True
    else:
        print('File with path={} is not existed'.format(path))
        return False


def read_cfg_file(root_dir=os.getcwd()):
    path = os.path.join(root_dir, 'cfg.json')
    with open(path, 'r') as fp:
        cfg_dict = json.load(fp)
        device = cfg_dict['target_device']
    return device


def read_log_file(log_dir, fname):
    data = str()

    path = os.path.join(log_dir, fname)
    if os.path.exists(path):
        with open(path, 'rt') as fp:
            data = fp.read()

    return data


def setup_compilers():
    patterns = {
        'flops': str(r'Flops: .+?, i\.e (?P<kmacpp>[\d\.]+) KMac \/ pxl'),
        'metrics': str(r'Results: (?P<codec>[\w\d\-\.]+)\s+(?P<metrics>[\s\d\.\-+ena]+)'),
        'name': str(r'(?P<name>.*)_(\d+)\.'),
        'total': str(r'TOTAL: (?P<hour>\d+):(?P<minute>\d+):(?P<second>[\d\.]+)'),
    }

    compilers = dict()
    for key, value in patterns.items():
        compilers[key] = re.compile(value)

    return compilers


def sort_filenames(enc_log_dir):
    filenames = os.listdir(enc_log_dir)
    filenames = sorted(filenames)
    return filenames


def format_codec_time(device, codec_time):
    # time string of codec: 'cpu_time gpu_time'
    if codec_time < 0:
        codec_time = 'None'
    if device == 'cpu':
        time_str = '\tNone\t{}'.format(codec_time)
    else:
        time_str = '\t{}\tNone'.format(codec_time)
    return time_str


def combine_codec_time(device, enc_time, dec_time):
    out_str = str()

    if dec_time is None:
        out_str += '\tNone\tNone'
    else:
        out_str += format_codec_time(device, dec_time)

    if enc_time is None:
        out_str += '\tNone\tNone'
    else:
        out_str += format_codec_time(device, enc_time)

    return out_str


def parse_codec_time(re_compile, has_data, log_data):
    codec_time = -1.0

    if not has_data:
        return codec_time

    try:
        this = re_compile.search(log_data)
        h_str, m_str, s_str = this.group('hour'), this.group('minute'), this.group('second')
        codec_time = 3600 * float(h_str) + 60 * float(m_str) + 1 * float(s_str)
    except:  # noqa: E722
        pass

    return codec_time


def parse_cur_name(reg_compilers, fname):
    this = reg_compilers['name'].search(fname)
    cur_fname = this.group('name')
    return cur_fname


def parse_enc_metrics(re_compiler, log_data, cur_fname):
    has_data = True

    if len(log_data) == 0:
        out_str = cur_fname
        return out_str, has_data

    try:
        this = re_compiler.search(log_data)
        codec_str = this.group('codec').strip()
        metrics_str = this.group('metrics').strip()
        out_str = f'{codec_str}\t{metrics_str}'
        has_data = True
    except:  # noqa: E722
        out_str = 'NO metrics'
        has_data = False

    return out_str, has_data


def parse_dec_flops(re_compiler, log_data, use_kmacpp):
    has_data = True

    if len(log_data) == 0:
        has_data = False
        out_str = '\tNone'
        return out_str, has_data

    if use_kmacpp:
        try:
            this = re_compiler.search(log_data)
            out_str = '\t{}'.format(this.group('kmacpp'))
        except:  # noqa: E722
            out_str = '\tNo decoder info'
            has_data = False
    else:
        out_str = '\tNone'

    return out_str, has_data


def parse_enc_log(reg_compilers, log_dir, fname, cur_fname):
    log_data = read_log_file(log_dir, fname)
    out_str, has_data = parse_enc_metrics(reg_compilers['metrics'], log_data, cur_fname)
    enc_time = parse_codec_time(reg_compilers['total'], has_data, log_data)
    return out_str, enc_time


def parse_dec_log(reg_compilers, log_dir, fname, use_kmacpp):
    log_data = read_log_file(log_dir, fname)
    out_str, has_data = parse_dec_flops(reg_compilers['flops'], log_data, use_kmacpp)
    seconds = parse_codec_time(reg_compilers['total'], has_data, log_data)
    return out_str, seconds


def parse_compare_log(log_dir, fname):
    out_str = str()

    log_data = read_log_file(log_dir, fname)
    if len(log_data) > 0:
        out_str = '\t{}'.format('match')
        if 'mismatch' in log_data.lower():
            out_str = '\t{}'.format('MISMATCH')

    return out_str


def write_pre_lines(fp, output_str, rate_iter, pre_lines):
    for _ in range(pre_lines):
        fp.write('{}\n'.format(output_str))
        rate_iter += 1
    return rate_iter


def write_last_line(fp, last_line, rate_iter, rate_num):
    for _ in range(rate_iter, rate_num):
        fp.write('{}\n'.format(last_line))


# ######################################################################################################################
#  main methods
# ######################################################################################################################
def main():
    args = parser_args()
    filenames = sort_filenames(args.enc_log)
    reg_compilers = setup_compilers()
    target_device = read_cfg_file(args.root_dir)

    with open(args.summary, 'w') as fp:
        act_fname = str()
        last_line = str()
        rate_iter = 0

        for fname in filenames:
            pre_lines = 0

            if not check_enc_log_file(args.enc_log, fname):
                continue

            cur_fname = parse_cur_name(reg_compilers, fname)

            # update states
            if act_fname != cur_fname:
                if len(act_fname) > 0:
                    write_last_line(fp, last_line, rate_iter, args.rate_num)

                act_fname = cur_fname
                pre_lines = args.pre_lines
                rate_iter = 1
            else:
                rate_iter += 1

            # parse log files
            output_str = str()

            out_str, enc_time = parse_enc_log(reg_compilers, args.enc_log, fname, cur_fname)
            output_str += out_str

            out_str, dec_time = parse_dec_log(reg_compilers, args.dec_log, fname, args.use_kmacpp)
            output_str += out_str

            output_str += combine_codec_time(target_device, enc_time, dec_time)

            output_str += parse_compare_log(args.cmp_log, fname)

            last_line = output_str

            rate_iter = write_pre_lines(fp, output_str, rate_iter, pre_lines)

            fp.write(output_str + '\n')

        write_last_line(fp, last_line, rate_iter, args.rate_num)


if __name__ == '__main__':
    main()
