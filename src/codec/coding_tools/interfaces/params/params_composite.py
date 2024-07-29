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

from src.codec.common import ArgParserDecorator

##
from .params_base import ParamsBase


class ParamsComposite(object):
    """Class for storing classes with parameters and wrapping their interface
    """

    __base_cls = None

    @property
    def base_cls(self) -> 'BaseModule':  # noqa: F821
        """ Return instance of holder

        Returns:
            BaseModule: base module, which is a holded of this instance
        """
        return self.__base_cls

    @base_cls.setter
    def base_cls(self, bc: 'BaseModule'):  # noqa: F821
        """Set instance of holder

        Args:
            bc (BaseModule): base module, which is a holded of this instance
        """
        self.__base_cls = bc

    def __init__(self, base: 'BaseModule', params_list: List[ParamsBase] = None):  # noqa: F821
        self._params_inst_list = params_list if params_list is not None else list()
        self.__base_cls = base

    def append(self, param: ParamsBase):
        """Add instance with set of parameters to internal list

        Args:
            param (ParamsBase): instance with set of parameters
        """
        self._params_inst_list.append(param)

    def remove_param_inst(self, param_inst_type):
        """Remove instance with specific set of parameters

        Args:
            param_inst_type: type of class with specific set of parameters
        """
        remove_list = []
        for i, inst in enumerate(self._params_inst_list):
            if isinstance(inst, param_inst_type):
                remove_list.append(i)
        for idx in reversed(remove_list):
            del self._params_inst_list[idx]

    def def_params_list(self, parser: ArgParserDecorator):
        """Define list of parameters in parser by calling function def_params_list of all sets of parameters in internal list

        Args:
            parser (ArgParserDecorator): parser of parameters
        """
        for p in self._params_inst_list:
            p.def_params_list(self.base_cls, parser)
            
    def set_defval2attrs(self) -> None:
        """Set default values from parameters to attributes
        """
        for p in self._params_inst_list:
            p.set_defval2attrs(self.base_cls)

    def load_params2attrs(self, **params):
        """Loading list of parameters to all sets of parameters in internal list
        """
        for p in self._params_inst_list:
            p.load_params2attrs(self.base_cls, **params)
            
    def signal_params_loaded(self) -> None:
        for p in self._params_inst_list:
            p.signal_params_loaded(self.base_cls)
        

    def store_attrs2dict(self) -> Dict[str, Any]:
        """Store parameters to dictionary

        Returns:
            Dict[str, Any]: dictionary with parameters
        """
        ans = dict()
        for p in self._params_inst_list:
            params = p.store_attrs2dict(self.base_cls)
            ans.update(params)
        return ans
    
    def get_params_inst_iter(self):
        """Iterate over instncies with parameters

        Yields:
            ParamBase: instance with a set of parameters
        """
        for p in self._params_inst_list:
            yield p

    def get_params_iter(self):
        """Iterate over all parameters

        Yields:
            name (str): name of parameter
            args (list): list with arguments of function add_argument
            kwargs (dict): dict with arguments of function add_argument
        """
        for p in self._params_inst_list:
            for n, a, k in p.get_params_iter():
                yield n, a, k

    def get_params_name_list(self) -> List[str]:
        """Get list with name of parameters

        Returns:
            List[str]: list with name of parameters
        """
        ans = []
        for p in self._params_inst_list:
            ans += p.get_params_name_list()
        return ans

    def add_missed_params(self, params: 'ParamsComposite'):
        """Added new parameters from 'params'

        Args:
            params (ParamsComposite): composer of parameters
        """
        cur_params_list = self.get_params_name_list()
        tmp_params_inst = ParamsBase()
        for n, a, k in params.get_params_iter():
            if n not in cur_params_list:
                tmp_params_inst.add_single_param(n, *a, **k)
        self.append(tmp_params_inst)

    # Service functions
    def load_params_from_owner(self, params_list: List[str]) -> bool:
        """
        Load parameters from owner.
        Return true if all parameters exists
        """
        ans = True
        parent = self.base_cls.owner
        params_dict = {}
        for p in params_list:
            if hasattr(parent, p):
                params_dict[p] = getattr(parent, p)
            else:
                ans = False

        for p in self._params_inst_list:
            p.load_params2attrs(self.base_cls, **params_dict)

        return ans
