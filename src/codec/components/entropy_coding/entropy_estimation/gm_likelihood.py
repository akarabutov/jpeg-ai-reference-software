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
import torch.nn.functional as F
import torch.optim


def simplified_softmax(x, dim):
    y = F.relu(x) + 1e-12
    return y / y.sum(dim, True)


def sum_lh(probs, lh):
    return sum([probs[:, i, :, :, :] * lh[i] for i in range(len(lh))])


def cal_y_likelihoods_decoupled(phi, psi, param_net, entropy, scale_hat, y=None, res=None):
    if not y == None:  # noqa: E711
        mean = param_net(torch.cat((psi, phi), dim=1))
        y_likelihoods = entropy(y, scale=scale_hat, mean=mean)
        return y_likelihoods

    y_likelihoods = entropy(res, scale=scale_hat, mean=torch.zeros_like(res))
    return y_likelihoods


def cal_y_likeli_Y(phi, psi, param_net, N_G, sym_flag, entropy, y_tilde):
    if sym_flag is False:
        if N_G == 3:
            prob0, mean0, scale_l0, scale_r0, prob1, mean1, \
                scale_l1, scale_r1, prob2, mean2, scale_l2, scale_r2 = \
                param_net(torch.cat((
                    phi.to(dtype=torch.float32),
                    psi.to(dtype=torch.float32)), dim=1)).chunk(12, dim=1)
            probs = torch.stack([prob0, prob1, prob2], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde,
                                     scale_l=scale_l0,
                                     scale_r=scale_r0,
                                     mean=mean0)
            y_likelihoods1 = entropy(y_tilde,
                                     scale_l=scale_l1,
                                     scale_r=scale_r1,
                                     mean=mean1)
            y_likelihoods2 = entropy(y_tilde,
                                     scale_l=scale_l2,
                                     scale_r=scale_r2,
                                     mean=mean2)
            y_lh = [y_likelihoods0, y_likelihoods1, y_likelihoods2]
            y_likelihoods = sum_lh(probs, y_lh)
        elif N_G == 2:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            tmp_res = param_net(param_net_args).chunk(8, dim=1)
            prob0, mean0, scale_l0, scale_r0, prob1, mean1, scale_l1, scale_r1 = tmp_res
            probs = torch.stack([prob0, prob1], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde,
                                     scale_l=scale_l0,
                                     scale_r=scale_r0,
                                     mean=mean0)
            y_likelihoods1 = entropy(y_tilde,
                                     scale_l=scale_l1,
                                     scale_r=scale_r1,
                                     mean=mean1)
            y_lh = [y_likelihoods0, y_likelihoods1]
            y_likelihoods = sum_lh(probs, y_lh)
        else:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            mean, scale_l, scale_r = param_net(param_net_args).chunk(3, dim=1)
            y_likelihoods = entropy(y_tilde,
                                    scale_l=scale_l,
                                    scale_r=scale_r,
                                    mean=mean)
    else:
        if N_G == 3:
            phi_psi = (phi.to(dtype=torch.float32), psi.to(dtype=torch.float32))
            param_net_args = torch.cat(phi_psi, dim=1)
            tmp_res = param_net(param_net_args).chunk(9, dim=1)
            prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = tmp_res

            probs = torch.stack([prob0, prob1, prob2], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde, scale=scale0, mean=mean0)
            y_likelihoods1 = entropy(y_tilde, scale=scale1, mean=mean1)
            y_likelihoods2 = entropy(y_tilde, scale=scale2, mean=mean2)
            y_lh = [y_likelihoods0, y_likelihoods1, y_likelihoods2]
            y_likelihoods = sum_lh(probs, y_lh)

        elif N_G == 2:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            tmp_res = param_net(param_net_args).chunk(6, dim=1)
            prob0, mean0, scale0, prob1, mean1, scale1 = tmp_res

            probs = torch.stack([prob0, prob1], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde, scale=scale0, mean=mean0)
            y_likelihoods1 = entropy(y_tilde, scale=scale1, mean=mean1)
            y_lh = [y_likelihoods0, y_likelihoods1]
            y_likelihoods = sum_lh(probs, y_lh)
        else:
            mean, scale = param_net(
                torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                          dim=1)).chunk(2, dim=1)

            y_likelihoods = entropy(y_tilde, scale=scale, mean=mean)

    return y_likelihoods


def cal_y_likeli_UV(phi, psi, param_net, N_G, sym_flag, entropy, y_tilde):
    if sym_flag is False:
        if N_G == 3:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            tmp_res = param_net(param_net_args).chunk(12, dim=1)
            prob0, mean0, scale_l0, scale_r0, prob1, mean1, scale_l1, scale_r1, prob2, mean2, scale_l2, scale_r2 = tmp_res

            probs = torch.stack([prob0, prob1, prob2], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde,
                                     scale_l=scale_l0,
                                     scale_r=scale_r0,
                                     mean=mean0)
            y_likelihoods1 = entropy(y_tilde,
                                     scale_l=scale_l1,
                                     scale_r=scale_r1,
                                     mean=mean1)
            y_likelihoods2 = entropy(y_tilde,
                                     scale_l=scale_l2,
                                     scale_r=scale_r2,
                                     mean=mean2)
            y_lh = [y_likelihoods0, y_likelihoods1, y_likelihoods2]
            y_likelihoods = sum_lh(probs, y_lh)

        elif N_G == 2:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            prob0, mean0, scale_l0, scale_r0, prob1, mean1, scale_l1, scale_r1 = param_net(
                param_net_args).chunk(8, dim=1)
            probs = torch.stack([prob0, prob1], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde,
                                     scale_l=scale_l0,
                                     scale_r=scale_r0,
                                     mean=mean0)
            y_likelihoods1 = entropy(y_tilde,
                                     scale_l=scale_l1,
                                     scale_r=scale_r1,
                                     mean=mean1)
            y_lh = [y_likelihoods0, y_likelihoods1]
            y_likelihoods = sum_lh(probs, y_lh)
        else:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            mean, scale_l, scale_r = param_net(param_net_args).chunk(3, dim=1)

            y_likelihoods = entropy(y_tilde,
                                    scale_l=scale_l,
                                    scale_r=scale_r,
                                    mean=mean)
    else:
        if N_G == 3:
            prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = param_net(
                torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                          dim=1)).chunk(9, dim=1)
            probs = torch.stack([prob0, prob1, prob2], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde, scale=scale0, mean=mean0)
            y_likelihoods1 = entropy(y_tilde, scale=scale1, mean=mean1)
            y_likelihoods2 = entropy(y_tilde, scale=scale2, mean=mean2)
            y_lh = [y_likelihoods0, y_likelihoods1, y_likelihoods2]
            y_likelihoods = sum_lh(probs, y_lh)

        elif N_G == 2:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            tmp_res = param_net(param_net_args).chunk(6, dim=1)
            prob0, mean0, scale0, prob1, mean1, scale1 = tmp_res

            probs = torch.stack([prob0, prob1], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde, scale=scale0, mean=mean0)
            y_likelihoods1 = entropy(y_tilde, scale=scale1, mean=mean1)
            y_lh = [y_likelihoods0, y_likelihoods1]
            y_likelihoods = sum_lh(probs, y_lh)
        else:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            mean, scale = param_net(param_net_args).chunk(2, dim=1)

            y_likelihoods = entropy(y_tilde, scale=scale, mean=mean)

    return y_likelihoods


def cal_y_likeli(phi, psi, param_net, N_G, sym_flag, entropy, y_tilde):
    if sym_flag is False:
        if N_G == 3:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            tmp_res = param_net(param_net_args).chunk(12, dim=1)
            prob0, mean0, scale_l0, scale_r0, prob1, mean1, scale_l1, scale_r1, prob2, mean2, scale_l2, scale_r2 = tmp_res

            probs = torch.stack([prob0, prob1, prob2], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde,
                                     scale_l=scale_l0,
                                     scale_r=scale_r0,
                                     mean=mean0)
            y_likelihoods1 = entropy(y_tilde,
                                     scale_l=scale_l1,
                                     scale_r=scale_r1,
                                     mean=mean1)
            y_likelihoods2 = entropy(y_tilde,
                                     scale_l=scale_l2,
                                     scale_r=scale_r2,
                                     mean=mean2)
            y_lh = [y_likelihoods0, y_likelihoods1, y_likelihoods2]
            y_likelihoods = sum_lh(probs, y_lh)

        elif N_G == 2:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            prob0, mean0, scale_l0, scale_r0, prob1, mean1, scale_l1, scale_r1 = param_net(
                param_net_args).chunk(8, dim=1)
            probs = torch.stack([prob0, prob1], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde,
                                     scale_l=scale_l0,
                                     scale_r=scale_r0,
                                     mean=mean0)
            y_likelihoods1 = entropy(y_tilde,
                                     scale_l=scale_l1,
                                     scale_r=scale_r1,
                                     mean=mean1)
            y_lh = [y_likelihoods0, y_likelihoods1]
            y_likelihoods = sum_lh(probs, y_lh)
        else:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            mean, scale_l, scale_r = param_net(param_net_args).chunk(3, dim=1)

            y_likelihoods = entropy(y_tilde,
                                    scale_l=scale_l,
                                    scale_r=scale_r,
                                    mean=mean)
    else:
        if N_G == 3:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = param_net(
                param_net_args).chunk(9, dim=1)
            probs = torch.stack([prob0, prob1, prob2], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde, scale=scale0, mean=mean0)
            y_likelihoods1 = entropy(y_tilde, scale=scale1, mean=mean1)
            y_likelihoods2 = entropy(y_tilde, scale=scale2, mean=mean2)
            y_lh = [y_likelihoods0, y_likelihoods1, y_likelihoods2]
            y_likelihoods = sum_lh(probs, y_lh)
        elif N_G == 2:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            prob0, mean0, scale0, prob1, mean1, scale1 = param_net(param_net_args).chunk(6, dim=1)
            probs = torch.stack([prob0, prob1], dim=1)
            probs = simplified_softmax(probs, dim=1)

            y_likelihoods0 = entropy(y_tilde, scale=scale0, mean=mean0)
            y_likelihoods1 = entropy(y_tilde, scale=scale1, mean=mean1)
            y_lh = [y_likelihoods0, y_likelihoods1]
            y_likelihoods = sum_lh(probs, y_lh)
        else:
            param_net_args = torch.cat((phi.to(dtype=torch.float32), psi.to(dtype=torch.float32)),
                                       dim=1)
            tmp_res = param_net(param_net_args).chunk(2, dim=1)
            mean, scale = tmp_res

            y_likelihoods = entropy(y_tilde, scale=scale, mean=mean)

    return y_likelihoods


def cal_likeli_gm(psi, entropy, y_tilde):
    mean, scale = psi.to(dtype=torch.float32).chunk(2, dim=1)
    y_likelihoods = entropy(y_tilde, scale=scale, mean=mean)
    return y_likelihoods
