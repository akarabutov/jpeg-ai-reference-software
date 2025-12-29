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
import os.path

import numpy as np
import torch

from src.codec.common import Image, Decisions
from src.codec.entropy_coding import create_lh_ecmodule

from ..interfaces import MultiToolsEngine, RdoPreProcInterface, ToolEngine
##
from .params import BitrateMatcherParams


class BitrateMatcher(RdoPreProcInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params_bm = BitrateMatcherParams()
        
    def bitrate_config(self):
        return os.path.join(self.bitrate_config_path, self.bitrate_config_name)

    def _params_loaded(self) -> None:
        if self.rewrite_beta_list_file and os.path.exists(self.bitrate_config()):
            os.remove(self.bitrate_config())
        self.rewrite_beta_list_file = 0
        
    def process_default(self, multitool: MultiToolsEngine) -> None:
        multitool.active_tool_idx = self.target_tool_idx
        multitool._set_beta_displacement_log(self.target_beta_disp_Y,  'Y')
        best_beta_dist_log_UV = self.target_beta_disp_UV
        if best_beta_dist_log_UV is None:
            best_beta_dist_log_UV = self.target_beta_disp_Y
        multitool._set_beta_displacement_log(best_beta_dist_log_UV, 'UV')        

    def process(self, multitool: MultiToolsEngine, ori_img: Image, *args, **kwargs):

        logger = self.logger

        if self.use_default == 1:
            best_beta_dist_log_Y, best_beta_dist_log_UV, best_tool_idx = self.skip_matching()
        else:
            if self.regen_beta_list:
                file_is_not_full = True
                best_beta_dist_log_Y, best_beta_dist_log_UV, best_tool_idx = None, None, 0
            else:
                best_beta_dist_log_Y, best_beta_dist_log_UV, best_tool_idx, file_is_not_full = self.read_from_file()
            if file_is_not_full:
                best_beta_dist_log_Y, best_tool_idx = self.match_luma(ori_img, multitool, *args, **kwargs)
                
                best_beta_dist_log_UV = best_beta_dist_log_Y if self.independent_beta_UV else self.find_UV_beta_with_hyperopt(ori_img,
                                                                   multitool.tools[best_tool_idx],
                                                                   best_beta_dist_log_Y)
                logger.debug(
                    f'best model is {best_tool_idx} best beta displacment in log domain for Y is {best_beta_dist_log_Y} and for UV is {best_beta_dist_log_UV}'
                )

                os.makedirs(self.bitrate_config_path, exist_ok=True)
                with open(self.bitrate_config(), 'a') as logfile:
                    line_id = self.get_owner_param('image_filepath').split('/')[-1] + '_' + str(
                        self.get_target_bpp_int() / 100)
                    line = f'{line_id} {best_tool_idx} {best_beta_dist_log_Y} {best_beta_dist_log_UV} \n'
                    logfile.write(line)
                    logfile.close()
        if self.independent_beta_UV:
            best_beta_dist_log_UV = best_beta_dist_log_Y
            
        #assert 0 < best_beta_dist_log_Y < 1 and 0 < best_beta_dist_log_UV < 1
        
        multitool.active_tool_idx = best_tool_idx
        multitool._set_beta_displacement_log(best_beta_dist_log_Y,  'Y')
        multitool._set_beta_displacement_log(best_beta_dist_log_UV, 'UV')
        
        self.beta_dist_log_Y = best_beta_dist_log_Y
        self.beta_dist_log_UV = best_beta_dist_log_UV

        logger.debug(f'beta y is {best_beta_dist_log_Y}')
        logger.debug(f'beta uv is {best_beta_dist_log_UV}')

    def match_luma(self, ori_img, multitool: MultiToolsEngine, *args, **kwargs):
        # First attempt: we search within beta ranges on which models were trained
        self.w = ori_img.shape[-1]
        self.h = ori_img.shape[-2]
        best_loss = np.inf
        best_unmatched_bits_mismatch = np.inf

        ### quick select tool_idx
        target_bits = self.get_target_bpp_int() / 100
        print(f"target bits: {target_bits}")
        bit_diff_max =np.inf

        for tool_idx in range(len(multitool.tools)):
            multitool.active_tool_idx = tool_idx
            decision = {}
            default_bits, _, _ = self._try_beta(0, target_bits, ori_img, decision, multitool.tools[tool_idx], *args, **kwargs)
            bit_diff = abs(default_bits - target_bits)/default_bits
            if bit_diff < bit_diff_max:
                bit_diff_max = bit_diff
                use_idx = tool_idx
            torch.cuda.empty_cache()

        multitool.active_tool_idx = use_idx
        self.beta_min, self.beta_max = multitool.tools[use_idx].BDL_range
        print("begin brm of the tool id ", use_idx)
        beta, loss, bits_mismatch = self.beta_linear_interpolation(ori_img, multitool.tools[use_idx])

        if loss < best_loss:
            best_beta = beta
            best_tool_idx = use_idx
            best_loss = loss

        elif abs(bits_mismatch) < best_unmatched_bits_mismatch:
            best_unmatched_bits_mismatch = abs(bits_mismatch)
            best_unmatched_beta = beta
            best_unmatched_tool_idx = use_idx


        if best_loss != np.inf:
            return best_beta, best_tool_idx

        self.logger.warn(
            f'Rate matching failed: best possible matching {100*best_unmatched_bits_mismatch.item():.1f}%'
        )

        return best_unmatched_beta, best_unmatched_tool_idx

    def read_from_file(self, *args, **kwargs):
        file_is_not_full = False
        best_beta = None
        best_beta_UV = None  # best beta is for all, best beta UV is only for UV
        best_tool_idx = None
        line_id = self.get_owner_param('image_filepath').split('/')[-1] + '_' + str(
            self.get_target_bpp())
        if not os.path.isfile(self.bitrate_config()):
            file_is_not_full = True
        if os.path.isfile(self.bitrate_config()):
            with open(self.bitrate_config(), 'r') as logfile:
                text = logfile.read()
                for line in text.split('\n'):
                    if len(line) > 0 and line.split()[0] == line_id:
                        best_beta_UV = float(line.split()[3])
                        best_beta = float(line.split()[2])
                        best_tool_idx = int(line.split()[1])
                if best_beta is None or best_beta_UV is None:
                    self.logger.info(
                        'You are trying to use beta list with does not correlate with selected desc.json and bpp set! Bitstream size tune is started, test will take longer time...'
                    )
                    file_is_not_full = True
        return best_beta, best_beta_UV, best_tool_idx, file_is_not_full

    def resize(self, ori_img: Image):
        scale = 2
        multiplier = 8
        new_ori_img = ori_img[:, :, ::scale, ::scale]
        self.w = (new_ori_img.shape[-1] // multiplier) * multiplier
        self.h = (new_ori_img.shape[-2] // multiplier) * multiplier
        return new_ori_img[:, :, :self.h, :self.w]

    def skip_matching(self):
        target_bits = self.get_target_bpp_int()
        try:
            idx = self.default_target_rates.index(target_bits)
            return self.default_beta_disp_log[idx], self.default_beta_disp_log[idx], self.default_models[idx]
        except:
            print('no beta can be found for selected target rate')

    def _try_beta(self, beta, target_bits, ori_img: Image,  decision: Decisions, tool: ToolEngine, *args,
                  **kwargs):
        tool.owner._set_beta_displacement_log(beta, 'Y')
        tool.owner._set_beta_displacement_log(beta, 'UV')

        decision = tool.compress(ori_img.clone(), decision, *args, **kwargs)
        ec = create_lh_ecmodule()
        tool.encode(ec, decision)
        total_bits = ec.get_total_bits()
        result_bits = total_bits / ori_img.shape[-1] / ori_img.shape[-2]
        bits_mismatch = (result_bits - target_bits) / target_bits

        if bits_mismatch <= self.tolerance_max and bits_mismatch >= (self.tolerance_min):
            # convert the original image to YUV format, just for metric calculation
            ori_img2 = ori_img.clone()
            ori_img2.to_YUV_()
            reco = tool.decompress(decision)
            reco.convert_range_(ori_img.data_range)
            criterion = torch.nn.MSELoss()
            distortion = criterion(reco.components['a'], ori_img2.components['a']) +\
                         criterion(reco.components['b'], ori_img2.components['b']) +\
                         criterion(reco.components['c'], ori_img2.components['c'])

            return result_bits, distortion, decision
        else:
            return result_bits, np.inf, decision

    
    def beta_linear_interpolation(self, ori_img: Image, tool: ToolEngine, *args, **kwargs):
        import math
        target_bits = self.get_target_bpp_int() / 100
        max_beta = self.beta_max
        min_beta = self.beta_min
        best_distortion = np.inf

        decision = {}

        ## speedup try_beta
        base_bits, base_distortion, decision = self._try_beta(0, target_bits, ori_img, decision, tool, *args, **kwargs)
        base_bits_mismatch = (base_bits - target_bits) / target_bits
        if base_bits_mismatch <= self.tolerance_max and base_bits_mismatch >= (self.tolerance_min):
            return 0, base_distortion, base_bits_mismatch
        min_bits, _, _ = self._try_beta(min_beta, target_bits, ori_img, decision, tool, *args, **kwargs)
        max_bits, _, _ = self._try_beta(max_beta, target_bits, ori_img, decision, tool, *args, **kwargs)
        if base_bits < target_bits:
            print("upper search")
            judge_condition, self.tolerance_max, min_beta, min_bits = 0, 0, 0, base_bits
            best_bits = np.inf
            
        else:
            print("down search")
            judge_condition, self.tolerance_min, max_beta, max_bits = 1, 0, 0, base_bits
            best_bits = -np.inf

            if target_bits == 0.12:
                self.tolerance_max = self.tolerance_max - 0.03
            else:
                self.tolerance_max = self.tolerance_max

        ### use linear interploation calculating the causuir target beta,
        now_beta = int(((((max_beta - min_beta) * (math.log(target_bits) - math.log(min_bits))) / (math.log(max_bits) - math.log(min_bits))) + min_beta))
        min_beta = now_beta-100
        max_beta = now_beta+100
        ## speedup try_beta
        #now_bits, now_distortion, _ = self._try_beta(now_beta, target_bits, ori_img, decision, tool, *args, **kwargs)

        #now_bits_mismatch = (now_bits - target_bits) / target_bits
        self.logger.debug(f" => current selected iter, beta = {now_beta}, min_beta = {min_beta}, max_beta={max_beta}")
        
        
        ### if max bit diff is  bigger than 0.1, need to finetune
        while min_beta <= max_beta:
            now_beta = min_beta + math.ceil((max_beta - min_beta) / 2)

            now_bits, now_distortion, _ = self._try_beta(now_beta, target_bits, ori_img, decision, tool, *args, **kwargs)
            now_bits_mismatch = (now_bits - target_bits) / target_bits
            self.logger.debug(f" => current selected iter, beta = {now_beta}, target_bits = {target_bits}, best_bits={best_bits}, bits = {now_bits},  mismatch = {now_bits_mismatch}, tol_min={self.tolerance_min}, tol_max={self.tolerance_max}, min_beta = {min_beta}, max_beta={max_beta}")

            if judge_condition == 0:
                if (now_bits_mismatch <= self.tolerance_max) and (now_bits_mismatch >= self.tolerance_min) and (now_bits < best_bits):
                    best_beta = now_beta
                    best_distortion = now_distortion
                    best_bits_mismatch = now_bits_mismatch
                    best_bits = now_bits
                search_bits_mismatch = (now_bits - target_bits*(1+self.tolerance_min)) / (target_bits*(1+self.tolerance_min))
            else:
                if (now_bits_mismatch <= self.tolerance_max) and (now_bits_mismatch >= self.tolerance_min) and (now_bits > best_bits):
                    best_beta = now_beta
                    best_distortion = now_distortion
                    best_bits_mismatch = now_bits_mismatch
                    best_bits = now_bits
                search_bits_mismatch = (now_bits - target_bits*(1+self.tolerance_max)) / (target_bits*(1+self.tolerance_max))

            if search_bits_mismatch > 0:
                max_beta = now_beta - 1
            else:
                min_beta = now_beta + 1
        
        if best_distortion == np.inf:
            print("Don't search good results")
            return now_beta, now_distortion, now_bits_mismatch
        else:
            print("Search good results")
            #print("initial linear interpolation ", first_now_beta, "best_beta ", best_beta, "beta gap ", (first_now_beta-best_beta)  )
            return best_beta, best_distortion, best_bits_mismatch


    def find_UV_beta_with_hyperopt(self, ori_img: Image, tool: ToolEngine, best_beta_Y, *args,
                                   **kwargs):
        from hyperopt import atpe, fmin, hp
        self.iter_number = 0
        self.best_check = 1e18
        self.best_args = None
        self.best_bitrate_mismatch = None
        self.best_distortion = None
        if best_beta_Y == 0:
            best_beta_Y = best_beta_Y - 0.01
        v1_UV = max(self.beta_min_mult*best_beta_Y, best_beta_Y * 0.1)
        v2_UV = min(self.beta_max_mult*best_beta_Y, best_beta_Y * 3)
        min_UV = min(v1_UV, v2_UV)
        max_UV = max(v1_UV, v2_UV)
        parameters_space = [best_beta_Y, hp.uniform('beta_UV', min_UV, max_UV), ori_img, tool]
        best = fmin(self.find_loss,
                    parameters_space,
                    algo=atpe.suggest,
                    max_evals=self.max_iterations_stage2,
                    show_progressbar=0)
        self.beta_UV = best['beta_UV']
        self.beta_Y = best_beta_Y
        self.beta_UV = int(np.round(self.beta_UV))
        torch.cuda.empty_cache()
        return self.beta_UV

    def find_loss(self, local_args):
        self.iter_number += 1
        self.img_format = 'yuv'
        weights = [0, 0, 10000]
        arg_list = [x for x in local_args]
        target_bits = self.get_target_bpp_int()/100

        criterion_mult = [0.8, 0.1, 0.1]

        self.beta_Y = arg_list[0]
        self.beta_UV = arg_list[1]
        ori_img = arg_list[2]
        tool = arg_list[3]

        self.beta_Y = np.round(self.beta_Y * 65535) / 65535  # to make it 16bits
        self.beta_UV = np.round(self.beta_UV * 65535) / 65535  # to make it 16bits
        
        tool.owner._set_beta_displacement_log(self.beta_Y, 'Y')
        tool.owner._set_beta_displacement_log(self.beta_UV, 'UV') # use the beta in the code

        decision = {}
        decision = tool.compress(ori_img.clone(), decision)
        ec = create_lh_ecmodule()
        tool.encode(ec, decision)
        reco = tool.decompress(decision)
        
        # convert the original image to YUV format, just for metric calculation
        ori_img2 = ori_img.clone()
        ori_img2.to_YUV_()
        reco = tool.decompress(decision)
        reco.convert_range_(ori_img.data_range)
    
        result_bits = ec.get_total_bits() / ori_img2.shape[-1] / ori_img2.shape[-2]
        distortion = 0

        range_diff = 1
        range_diff2 = range_diff * range_diff
        MultiplierMSSSIM = 1e7 

        distortion = self._calculate_mssim_distortion(reco.components['a'],ori_img2.components['a'],ch_weight=criterion_mult[0], multiplierMSSIM=MultiplierMSSSIM) + \
            self._calculate_mssim_distortion(reco.components['b'],ori_img2.components['b'],ch_weight=criterion_mult[1], multiplierMSSIM=MultiplierMSSSIM)+ \
            self._calculate_mssim_distortion(reco.components['c'],ori_img2.components['c'],ch_weight=criterion_mult[2], multiplierMSSIM=MultiplierMSSSIM)

        bits_mismatch = ((result_bits - target_bits) / target_bits)
        overbit = max((bits_mismatch - self.tolerance_max) / abs(self.tolerance_max + 1E-10), 1)
        underbit = max((self.tolerance_min - bits_mismatch) / abs(self.tolerance_min + 1E-10), 1)
        loss_overbit = overbit - 1
        loss_underbit = underbit - 1
        loss_distortion = distortion.item()
        wm = [loss_overbit, loss_underbit, (10*loss_distortion * ((self.beta_Y + 2048)/2048)* tool.base_model_beta + result_bits)] #using smaller beta for trade off
        loss = sum([w1 * w2 for w1, w2 in zip(weights, wm)])

        if loss < self.best_check and bits_mismatch < 0.09:
            self.best_check = loss.item()
            self.best_args = [self.beta_Y, self.beta_UV]
            self.best_bitrate_mismatch = bits_mismatch
            self.best_distortion = distortion.item()
            self.logger.info(
                f'bpp={target_bits}, loss={loss} distortion={distortion.item()} bits_mismatch={bits_mismatch} beta_Y={self.beta_Y} beta_UV={self.beta_UV} i={self.iter_number}'
            )

        return self.best_check

    def _calculate_mssim_distortion(self, rec, orig, ch_weight, multiplierMSSIM):
        criterion = {}
        criterion['mse'] = torch.nn.MSELoss()
        criterion['msssim'] = MSSSIM()
        mssim = criterion['msssim'](orig, rec) * ch_weight * multiplierMSSIM
        mse = criterion['mse'](orig, rec) * ch_weight * 5000
        self.MSSSIMweight = 0
        mssim = self.MSSSIMweight * mssim + (1.0 - self.MSSSIMweight) * mse
        return mssim

