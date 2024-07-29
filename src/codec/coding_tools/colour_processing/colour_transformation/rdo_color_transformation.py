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

import torch
import torch.nn as nn

from src.codec.common import ColorSpace
from src.codec.entropy_coding import ECLibLH, ECModule
from src.codec.metrics import MSSSIM

from ..interfaces import ToolEngine
##
from .params import RdoColorTransformParams


class RDOColorTrasformation(ToolEngine):
    def __init__(self, **kwargs):
        super(RDOColorTrasformation, self).__init__(has_enabled_flag=False, **kwargs)
        self._params_colour_transform = RdoColorTransformParams()

    def find_best_colour_transform(self, img, tool, beta):
        if not self.use:
            return None
        self.beta = beta
        torch.manual_seed(0)
        tool.use_colour_transform = 1
        epochs = 10
        '''
        if beta> 0.005:
            lr = 0.02
        else:
            lr = 0.0001'''

        lr = 0.02
        matrix = torch.nn.Parameter(torch.eye(3, device=img.device))
        offset = torch.nn.Parameter(torch.zeros(3, device=img.device))
        best_matrix = torch.eye(3, device=img.device)
        best_offset = torch.zeros(3, device=img.device)
        params_list = [{'params': matrix}, {'params': offset}]
        optimizer = torch.optim.Adam(params_list, slr=lr)
        best_loss = 1e9
        tool.inverse_matrix = best_matrix
        tool.colour_transform_offset = best_offset
        loss = self.find_loss(self.resize(img), tool, beta)
        if loss < best_loss:
            best_matrix = matrix.detach().clone()
            best_offset = offset.detach().clone()
            best_loss = loss.detach().clone()
        print('loss before =', loss * 1000)

        with torch.set_grad_enabled(True):
            with torch.enable_grad():
                tool.train()
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    tool.inverse_matrix = matrix * 1
                    tool.colour_transform_offset = offset * 1
                    loss = self.find_loss(self.resize(img), tool, beta)
                    if loss < best_loss:
                        best_matrix = matrix.clone()
                        best_offset = offset.clone()
                        best_loss = loss.clone()
                        print('epoch=', epoch, 'loss=', loss * 1000)
                    loss.backward()
                    optimizer.step()
        best_matrix = (torch.round(best_matrix * 100 + 100) - 100) / 100
        best_offset = (torch.round(best_offset * 100 + 100) - 100) / 100
        tool.inverse_matrix = best_matrix
        tool.colour_transform_offset = best_offset
        print(best_matrix)
        print(best_offset)
        print(beta)

        tool.eval()
        print('done!')

    def find_loss(self, img, tool, beta):
        self.img_format = 'yuv'

        criterion_mult = [0.8, 0.1, 0.1]

        ori_parent = tool.owner
        tool.set_parent(self)

        decision = {}
        decision = tool.compress(img, decision)
        ac = ECLibLH(None)
        ec = ECModule(ac)
        ac.encode_init()
        tool.encode(ec, decision, h=img.shape[-2], w=img.shape[-1])
        reco = tool.decompress(decision)
        reco = torch.clamp(reco, 0, 1)

        result_bits = ac.get_total_bits() / img.shape[-1] / img.shape[-2]

        reco_p = self._preprocess_img(reco)
        img_p = self._preprocess_img(img)

        # define loss function (criterion) and optimizer
        criterion = {}
        criterion['mse'] = nn.MSELoss()
        criterion['msssim'] = MSSSIM()

        xY = img_p[:, 0:1]
        xY_reco = reco_p[:, 0:1]
        xCbCr = img_p[:, 1:]
        xU_reco = reco_p[:, 1:2]
        xV_reco = reco_p[:, 2:]

        mse_Y_loss = criterion['mse'](xY, xY_reco)
        mse_Cb_loss = criterion['mse'](xCbCr[:, :1, :, :], xU_reco)
        mse_Cr_loss = criterion['mse'](xCbCr[:, 1:, :, :], xV_reco)

        msssim_Y_loss = criterion['msssim'](xY, xY_reco)
        msssim_Cb_loss = criterion['msssim'](xCbCr[:, :1, :, :], xU_reco)
        msssim_Cr_loss = criterion['msssim'](xCbCr[:, 1:, :, :], xV_reco)

        distortion_Y_loss = (
            1 - self.msssim_weight) * mse_Y_loss + self.msssim_weight * 1000 * msssim_Y_loss
        distortion_Cb_loss = (
            1 - self.msssim_weight) * mse_Cb_loss + self.msssim_weight * 1000 * msssim_Cb_loss
        distortion_Cr_loss = (
            1 - self.msssim_weight) * mse_Cr_loss + self.msssim_weight * 1000 * msssim_Cr_loss

        loss = self.beta * (distortion_Y_loss * criterion_mult[0] +
                            distortion_Cb_loss * criterion_mult[1] +
                            distortion_Cr_loss * criterion_mult[2]) + result_bits

        tool.set_parent(ori_parent)
        return loss

    def find_loss_old(self, img, tool, beta):
        self.img_format = 'yuv'

        criterion_mult = [0.8, 0.1, 0.1]

        ori_parent = tool.owner
        tool.set_parent(self)

        decision = {}
        decision = tool.compress(img, decision)
        ac = ECLibLH(None)
        ec = ECModule(ac)
        ac.encode_init()
        tool.encode(ec, decision, h=img.shape[-2], w=img.shape[-1])
        reco = tool.decompress(decision)

        result_bits = ac.get_total_bits() / img.shape[-1] / img.shape[-2]
        distortion = 0
        MultiplierMSSSIM = 310 * 255 * 255 / (max(self.get_img_norm()) - min(
            self.get_img_norm())) / (max(self.get_img_norm()) - min(self.get_img_norm()))
        multiplier = 0.1 * 255 * 255 / (max(self.get_img_norm()) - min(self.get_img_norm())) / (
            max(self.get_img_norm()) - min(self.get_img_norm()))
        reco_p = self._preprocess_img(reco)
        img_p = self._preprocess_img(img)
        for plane in range(img.shape[1]):
            distortion += self._calculate_mssim_distortion(reco_p[:, plane:plane + 1],
                                                           img_p[:, plane:plane + 1],
                                                           ch_weight=criterion_mult[plane],
                                                           multiplierMSSIM=MultiplierMSSSIM,
                                                           multiplier=multiplier)

        loss = distortion * self.beta + result_bits

        tool.set_parent(ori_parent)
        return loss

    def get_processed_img_shape(self):
        return self.h, self.w

    def resize(self, img):
        scale = 2
        multiplier = self.size_downscaler
        new_img = img[:, :, ::scale, ::scale]
        self.w = (new_img.shape[-1] // multiplier) * multiplier
        self.h = (new_img.shape[-2] // multiplier) * multiplier
        self.w = (self.w // 128) * 128
        self.h = (self.h // 128) * 128
        return new_img[:, :, :self.h, :self.w]

    def _calculate_mssim_distortion(self, rec, orig, ch_weight, multiplierMSSIM, multiplier):
        criterion = {}
        criterion['mse'] = torch.nn.MSELoss()
        criterion['msssim'] = MSSSIM()
        mssim = criterion['msssim'](orig, rec) * ch_weight * multiplierMSSIM
        mse = criterion['mse'](orig, rec) * ch_weight * multiplier
        self.MSSSIMweight = 0.1
        mssim = self.MSSSIMweight * mssim + (1.0 - self.MSSSIMweight) * mse
        return mssim

    def _preprocess_img(self, img):
        return ColorSpace.rgb_to_yuv(img).mul(255)
