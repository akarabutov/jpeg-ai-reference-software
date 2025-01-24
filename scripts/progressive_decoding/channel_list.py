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
import numpy
def get_sorted_channels(model_name):
    common_model_path = f"../VM_common_int/{model_name}.pth"
    state_dict = torch.load(common_model_path)
    channel_wise_entropy = state_dict['channel_wise_entropy'].detach().cpu().numpy()
    print(channel_wise_entropy)
    if numpy.sum(abs(channel_wise_entropy)) < 1e-5:
        ret = list(range(0, channel_wise_entropy.shape[0]))
    else:
        ret = (numpy.argsort(channel_wise_entropy)[::-1]).tolist()
    print(ret)
    return ret

sorted_channel_ids_dict = {

  "Y_0.075": get_sorted_channels("Y_0.075"),
  "Y_0.5": get_sorted_channels("Y_0.5"),
  "Y_0.002": get_sorted_channels("Y_0.002"),
  "Y_0.012": get_sorted_channels("Y_0.012"),

  "UV_0.075": get_sorted_channels("UV_0.075"),  
  "UV_0.5":  get_sorted_channels("UV_0.5"),
  "UV_0.002": get_sorted_channels("UV_0.002"),  
  "UV_0.012": get_sorted_channels("UV_0.012")

}
