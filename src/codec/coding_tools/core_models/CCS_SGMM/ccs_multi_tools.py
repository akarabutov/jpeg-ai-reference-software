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

from src.codec.coding_tools.interfaces import MultiToolsEngine
from ..base import CoreModelBase
##
from .ccs_sgmm_tool import CcsGvaeSGMM


class CcsGvaeFactory:
    """Factory for creation of new instance of CcsGvaeSGMM
    """
    @staticmethod
    def create_instance(*args, **kwargs):
        return CcsGvaeSGMM(*args, **kwargs)


class CcsGvaeMultiTools(MultiToolsEngine, CoreModelBase):
    """Multimodel instance for CcsGvaeSGMM.
    It contains models with all possible checkpoints.
    """
    def __init__(self, *args, **kwargs):
        super(CcsGvaeMultiTools, self).__init__(Ntools_max=6,
                                                factory=CcsGvaeFactory(),
                                                max_models_count=16, 
                                                *args,
                                                **kwargs)
        
    def get_base_model_beta(self):
        return self.get_active_tool().base_model_beta

            
    def _set_beta_displacement_log(self, beta_displacement_log: int, component):
        if component == 'Y':
            self.beta_displacement_log_Y = beta_displacement_log
        elif component == 'UV':
            self.beta_displacement_log_UV = beta_displacement_log
        else:
            assert(f'invalid component {component}')

    def get_alignment_size(self) -> int:
        """Get aligment size of the active model

        Returns:
            int: alignment size
        """
        return self.get_active_tool().get_alignment_size()
