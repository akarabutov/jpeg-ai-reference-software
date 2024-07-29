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

from ...coding_tools.interfaces.params import ParamsBase


class ParamsEC(ParamsBase):
    def __init__(self, *args, **kwargs):
        super(ParamsEC, self).__init__(*args, **kwargs)

        add_single_param = self.add_single_param

        types = kwargs.get('types', [])

        add_single_param('type', default=types[0], choices=types)
        add_single_param('debug', type=int, default=0)
        add_single_param('debug_start', type=int, default=0)
        add_single_param('verbose', type=int, default=0)
        add_single_param('rebuild_ae_cache', type=int, default=0)
                                                                            # PIH    TOH    SOQ        SOZ       SORp      SORs      UDI         RDI
        add_single_param('max_compressed_size', type=int, nargs="+", default=[10000, 10000, 160000000, 10000000, 80000000, 80000000, 2000000000, 160000000 ])
        add_single_param('quant_start', type=int, default=0.11)
        add_single_param('quant_end', type=int, default=54.82)
        add_single_param('quant_count', type=int, default=32)
        add_single_param('reverse_encode_order', type=int, default=1, choices=[0,1])
