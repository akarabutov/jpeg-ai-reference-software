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
from pathlib import Path

import pandas as pd
import torch


def parser_args():
    this = argparse.ArgumentParser('Tool for collecting results')

    this.add_argument('summary_path', type=str, help='Path to symmary.txt')
    this.add_argument('--output_path',
                      type=str,
                      default=os.getcwd(),
                      help='Path to root folder of project')
    this.add_argument('-output_csv',
                      default=False,
                      action='store_true',
                      help='Add additional data to summary txt which is used for RDLR parameters.')

    return this


def read_summary(fn):
    no_rdlr_sim_summary = pd.read_csv(fn,
                                      sep='\t',
                                      names=[
                                          'name',
                                          'bpp',
                                          'msssim_Torch',
                                          'msssim_iqa',
                                          'psnrY',
                                          'psnrU',
                                          'psnrV',
                                          'vif',
                                          'fsim',
                                          'nlpd',
                                          'iw_ssim',
                                          'vmaf',
                                          'psnrHVS',
                                          'kMAC_pxl',
                                          'DecGPU',
                                          'DecCPU',
                                          'EncGPU',
                                          'EncCPU',
                                          'match',
                                      ])
    no_rdlr_sim_summary['seq_name'] = no_rdlr_sim_summary.name.str.extract(
        r'VM_(.*)_8bit_sRGB_\d\d\d.png')
    no_rdlr_sim_summary['target_bpp'] = no_rdlr_sim_summary.name.str.extract(
        r'.*(\d\d\d).png').astype('int')

    # remove columns not needed for further processing
    no_rdlr_sim_summary = no_rdlr_sim_summary[[
        'name', 'target_bpp', 'bpp', 'msssim_Torch', 'psnrY', 'psnrU', 'psnrV', 'seq_name'
    ]]

    return no_rdlr_sim_summary


def get_images_list(no_rdlr_sim_summary):
    return no_rdlr_sim_summary.seq_name.unique()


def calc_slops(no_rdlr_sim_summary, images):
    slopes_df = []

    metrics = [
        'msssim_Torch',
        'psnrY',
        'psnrU',
        'psnrV',
    ]

    for idx_y, image in enumerate(images):
        current_seq_noRDLR_rate_points_df = no_rdlr_sim_summary.loc[
            no_rdlr_sim_summary.seq_name.str.contains(image), :].reset_index(drop=True).copy()
        current_seq_noRDLR_rate_points_df.drop_duplicates(inplace=True)
        current_seq_noRDLR_rate_points_df.sort_values(by='target_bpp',
                                                      inplace=True,
                                                      ascending=True)  # ensure points are in order
        current_seq_noRDLR_rate_points_df.reset_index(drop=True, inplace=True)

        for metric in metrics:
            # interpolate
            rate = torch.Tensor(current_seq_noRDLR_rate_points_df.bpp)
            rate_log = rate.log()

            metric_vals = torch.Tensor(current_seq_noRDLR_rate_points_df[metric])

            # calculate slopes
            gradient = torch.gradient(metric_vals, spacing=(rate_log, ))[0]
            for idx_rp in range(len(rate_log)):
                slope3 = torch.atan(gradient[idx_rp])
                slope3 = float(slope3 / torch.pi * 180)

                rp_df = pd.DataFrame({
                    'name': [current_seq_noRDLR_rate_points_df.name[idx_rp]],
                    'seq_name': [current_seq_noRDLR_rate_points_df.seq_name[idx_rp]],
                    'bpp': [current_seq_noRDLR_rate_points_df.bpp[idx_rp]],
                    'target_bpp': [current_seq_noRDLR_rate_points_df.target_bpp[idx_rp]],
                    'metric': [metric],
                    'slope': [slope3],
                })
                slopes_df.append(rp_df)

    slopes_df = pd.concat(slopes_df, ignore_index=True)
    return slopes_df


def write_slops(path_to_root, slopes_df):
    per_image_per_bpp_dir = Path(os.path.join(path_to_root, 'cfg', 'per-image-per-bpp'))
    seq_names = slopes_df.seq_name.unique()
    bpps = slopes_df.bpp.unique()

    for seq_name in seq_names:
        seq_dir = per_image_per_bpp_dir / seq_name
        seq_dir.mkdir(exist_ok=True)

        for bpp in bpps:
            per_image_per_bpp_cfg = seq_dir / f'bpp{bpp}.json'

            per_image_per_bpp_df = slopes_df[(slopes_df.bpp == bpp)
                                             & (slopes_df.seq_name == seq_name)]
            per_image_per_bpp_slopes = per_image_per_bpp_df.pivot_table(
                index='metric', values='slope').transpose().loc['slope', :]

            cfg_dict = {
                'model': {
                    'rdlr': {
                        'lossTypeBDcurveSlope_psnrY_slope': per_image_per_bpp_slopes.psnrY,
                        'lossTypeBDcurveSlope_psnrU_slope': per_image_per_bpp_slopes.psnrU,
                        'lossTypeBDcurveSlope_psnrV_slope': per_image_per_bpp_slopes.psnrV,
                        'lossTypeBDcurveSlope_msssimY_slope':
                        per_image_per_bpp_slopes.msssim_Torch,
                    }
                }
            }
            with open(per_image_per_bpp_cfg, 'w') as outfile:
                json.dump(cfg_dict, outfile, indent='\t')


def main():
    parser = parser_args()
    args, _ = parser.parse_known_args()

    no_rdlr_sim_summary = read_summary(args.summary_path)
    images_list = get_images_list(no_rdlr_sim_summary)

    slops = calc_slops(no_rdlr_sim_summary, images_list)

    if args.output_csv:
        slops.to_csv('bdcurve_slopes2.csv')
    else:
        write_slops(args.output_path, slops)

    print('means_per_metric_and_qp:')
    means_per_metric_and_qp = slops.pivot_table(index=['metric', 'target_bpp'], values='slope')
    print(means_per_metric_and_qp, '\n')

    print('means_per_metric:')
    means_per_metric = slops.pivot_table(index=['metric'], values='slope')
    print(means_per_metric)


if __name__ == '__main__':
    main()
