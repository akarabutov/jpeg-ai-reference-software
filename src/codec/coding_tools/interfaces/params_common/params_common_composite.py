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

from typing import Any, Dict, List

##
from ..params.params_base import ParamsBase
from ..params.params_composite import ParamsComposite


class ParamsCommonComposite(ParamsComposite):
    """Class for storing classes with parameters and wrapping their interface
    """

    @property
    def target_objs(self) -> List['BaseModule']:  # noqa: F821
        """ Return instance of holder

        Returns:
            BaseModule: base module, which is a holded of this instance
        """
        return self.base_cls.target_objs       

    def __init__(self, base: 'ParamsCommonObj', params_list: List[ParamsBase] = None):  # noqa: F821
        super(ParamsCommonComposite, self).__init__(base, params_list)
        for p in self.target_objs[0].params.get_params_inst_iter():
            self.append(p)

    def load_params2attrs(self, **params):
        """Loading list of parameters to all sets of parameters in internal list
        """
        if len(params) == 0:
            return
        for p in self._params_inst_list:
            p.load_params2attrs(self.base_cls, **params)
            for to in self.target_objs:
                to.load_params2attrs_recursively(**params)

    def store_attrs2dict(self) -> Dict[str, Any]:
        """Store parameters to dictionary

        Returns:
            Dict[str, Any]: dictionary with parameters
        """
        ans = dict()
        # Don't collect the parameters, because only real tools should store them
        return ans