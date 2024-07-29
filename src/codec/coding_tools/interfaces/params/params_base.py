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
from src.codec.common import ArgParserDecorator


class ParamsBase(object):
    def __init__(self, *args, **kwargs):
        self._params_list = list()  # List of parameters
        self.owner = None

    def add_single_param(self, name, *args, **kwargs) -> None:
        self._params_list.append({
            'name': name,
            'args': args,
            'kwargs': kwargs,
            'def': kwargs.get('default', None)
        })

    def get_params_name_list(self) -> List[str]:
        ans = []
        for params in self._params_list:
            ans.append(params['name'])
        return ans

    def def_params_list(self, base: 'BaseModule', parser: ArgParserDecorator):  # noqa: F821
        """

        Args:
            parser (ArgParserDecorator): decorator for parser
        """

        # Put all of our parameters to parser
        for param in self._params_list:
            parser.add_argument(param['name'], *param['args'], **param['kwargs'])

        # Create members in base with corresponding names
        for n, v in parser.get_section_params().items():
            setattr(base, n, v)
            
    def set_defval2attrs(self, base: 'BaseModule') -> None:  # noqa: F821
        """Setting default values for attributes

        Args:
            base (BaseModule): base class where we will store the parameters
        """
        for param in self._params_list:
            n = param['name']
            if 'def' in param:
                dv = param['def']
                dtype = param.get('kwargs', dict()).get('type', str)
                if isinstance(dv, list):
                    dv = [dtype(x) for x in dv]
                elif dv is None or isinstance(dv, dict):
                    pass
                else:
                    dv = dtype(dv)
                setattr(base, n, dv)        

    def load_params2attrs(self, base: 'BaseModule', **params) -> None:  # noqa: F821
        """Load parameters from dictionary

        Args:
            base (BaseModule): base class where we will store the parameters
        """
        for param in self._params_list:
            n = param['name']
            #def_val = param['def']

            cur_dv = getattr(base, n)
            dv = params.get(n, cur_dv)
            dtype = param.get('kwargs', dict()).get('type', str)
            if isinstance(dv, list):
                dv = [dtype(x) for x in dv]
            elif dv is None or isinstance(dv, dict):
                pass
            else:
                dv = dtype(dv)
            setattr(base, n, dv)

    def signal_params_loaded(self, base: 'BaseModule') -> None:
        # Handle 'inh' case
        for param in self._params_list:
            n = param['name']
            cur_dv = getattr(base, n)
            if cur_dv == 'inh':
                if base.has_owner():
                    dv = base.owner.get_owner_param(n)
                elif 'choices' in list(param['kwargs'].keys()):
                    dv = param['kwargs']['choices'][0]
                else:
                    raise NotImplementedError

                dtype = param.get('kwargs', dict()).get('type', str)
                if isinstance(dv, list):
                    dv = [dtype(x) for x in dv]
                elif dv is None or isinstance(dv, dict):
                    pass
                else:
                    dv = dtype(dv)
                setattr(base, n, dv)        
        self._params_loaded(base)

    def store_attrs2dict(self, base: 'BaseModule') -> Dict[str, Any]:  # noqa: F821
        """Store parameters to dictionary

        Returns:
            Dict[str, Any]: dictionary with parameters
        """
        ans = dict()
        for param in self._params_list:
            n = param['name']
            dv = getattr(base, n)
            ans[n] = dv
        return ans

    def get_params_iter(self):
        """Iterate over all parameters

        Yields:
            name (str): name of parameter
            args (list): list with arguments of function add_argument
            kwargs (dict): dict with arguments of function add_argument
        """
        for param in self._params_list:
            n = param['name']
            a = param['args']
            k = param['kwargs']
            yield n, a, k

    # Virtual functions
    def _params_loaded(self, base: 'BaseModule') -> None:  # noqa: F821
        """
        Function executes after module's parameters are loaded
        """
        pass
