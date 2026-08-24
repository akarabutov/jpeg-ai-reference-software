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

__all__ = ['init_optimizer']


def init_optimizer(args, net):

    analysis_models, synthesis_models, entropy_part_models, gain_unit_models = \
            net.parameters()

    opt_choices = {'adam': torch.optim.Adam}

    trained_models_list = list()
    trained_models_list_rate = list()

    if 'gain_unit' not in args.frozen_part:
        trained_models_list += gain_unit_models
        trained_models_list_rate += gain_unit_models
    if 'analysis' not in args.frozen_part:
        trained_models_list += analysis_models
        trained_models_list_rate += analysis_models
    if 'synthesis' not in args.frozen_part:
        trained_models_list += synthesis_models
    if 'entropy' not in args.frozen_part:
        trained_models_list += entropy_part_models
        trained_models_list_rate += entropy_part_models

    assert not args.zero_redundancy_optimizer

    optimizer = opt_choices[args.opt_type](trained_models_list,
                                            lr=args.lr,
                                            weight_decay=args.wd)

    return optimizer
