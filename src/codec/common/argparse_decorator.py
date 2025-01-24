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
import argparse
import json
import sys
from typing import Dict, List, Union, Tuple

from .utils import param_to_dict, update_dict_recursively
from collections import OrderedDict


class ArgParserDecorator:
    """
    Decorator for argparse, which allows to generate pyramid-like list of argumemnts and parse them

    For example, you may organize your arguments in the following way:
    -arg1
    -arg2
    -module1.arg1
    -module1.arg2
    -module2.arg1
    -module2.arg2

    by code:
    parser_ori = argparse.ArgumentParser()
    parser = ArgParserDecorator(parser_ori)
    parser.add_argument("-arg1")
    parser.add_argument("-arg2")
    sub_module1 = parser.add_sub_section_parser("module1")
    sub_module1.add_argument("arg1")
    sub_module1.add_argument("arg2")
    sub_module2 = parser.add_sub_section_parser("module2")
    sub_module2.add_argument("arg1")
    sub_module2.add_argument("arg2")

    Parsing of the arguments:
    args = parser.parse_args()

    It will return namespace with "arg1" and "arg2".
    To get access to other arguments, you may use
    `
    sub_args = parser.get_params()
    `

    """
    def __init__(self, base_parser: argparse.ArgumentParser, prefix=''):
        self.parser = base_parser
        self.param_prefix = prefix
        self.args_notsec = {}
        self.sec_parsers = {}

    def add_argument(self, *args, **kwargs) -> None:
        """Add argument to command line.
            Arguments may be stored to an internal dictionary into pyromidial structure or not.
            If args[0] start with "-" it will be interpretable as argument, which shouldn't be stored to dict. It may happends only on the first layer of pyrimid.
            Otherwise it will be stored to dictionary.
        """
        store_to_dict = True
        if len(self.param_prefix) > 0:
            assert not args[0].startswith('-')

        if len(self.param_prefix) > 0:
            arg_name = f'-{self.param_prefix}.{args[0]}'
        else:
            if args[0].startswith('-'):
                arg_name = args[0]
                store_to_dict = False
            else:
                arg_name = f'-{args[0]}'
        self.parser.add_argument(arg_name, *args[1:], **kwargs)
        if store_to_dict:
            self.args_notsec[args[0]] = kwargs.get('default', None)

    def get_params(self) -> Dict[str, None]:
        """Get dictionary with pyramid structure of arguments and their values

        Returns:
            _type_: _description_
        """
        from addict import Dict
        ans = Dict(self.args_notsec)
        for n, v in self.sec_parsers.items():
            ans[n] = v.get_params()
        return ans

    def get_section_params(self) -> Dict[str, None]:
        """Get dictionary with structure of arguments and their values of the current sub section

        Returns:
            _type_: _description_
        """
        ans = self.args_notsec
        return ans

    def add_sub_section_parser(self, name: str) -> 'ArgParserDecorator':
        new_name = name
        if len(self.param_prefix) > 0:
            new_name = f'{self.param_prefix}.{name}'
        ans = ArgParserDecorator(self.parser, new_name)
        self.sec_parsers[name] = ans
        return ans

    def parse_known_args(self, args=None, namespace=None) -> None:

        ans_known_args, ans_unknown_args = self.parser.parse_known_args(args, namespace)
        ans_known_args_dict = vars(ans_known_args)
        ans_known_args_pyramid_dict = self._convert_params_to_pyramid_dict(**ans_known_args_dict)
        self._load_params_from_pyramid_dict(**ans_known_args_pyramid_dict)
        ans_known_args_dict_clean = self.remove_params(ans_known_args_dict)
        return argparse.Namespace(**ans_known_args_dict_clean), ans_unknown_args

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        ans_args, _ = self.parse_known_args(args, namespace)
        return ans_args

    def load_params_from_cmd_line(self, args=None, namespace=None) -> Dict:
        ans_known_args, ans_unknown_args = self.parser.parse_known_args(args, namespace)
        ans_known_args = vars(ans_known_args)
        ans = dict()
        args = sys.argv if args is None else args
        for arg in ans_known_args.keys():
            if ans_known_args[arg] is None:
                continue
            if f'-{arg}' in args:
                d = param_to_dict(arg, ans_known_args[arg])
                ans = update_dict_recursively(ans, d)
        values = list()
        cur_key = None
        for arg in ans_unknown_args:
            if arg[0] == '-':
                if cur_key is not None:
                    d = param_to_dict(cur_key, values if len(values) > 1 else values[0])
                    ans = update_dict_recursively(ans, d)
                cur_key = arg[1:] if arg[1] != '-' else  None
                values = list()
            else:
                values.append(arg)
        if cur_key is not None:
            d = param_to_dict(cur_key, values if len(values) > 1 else values[0])
            ans = update_dict_recursively(ans, d)
        return ans
    
    def get_cfgs(self, cfg_paths: Union[str, List[str]]) -> List[str]:
        class LastUpdatedOrderedDict(OrderedDict):
            'Store items in the order the keys were last added'

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                self.move_to_end(key)
                
        def rec_load_cfgs(cfg_path: str) -> Tuple[Dict, List[str]]:
            ans_dict = LastUpdatedOrderedDict()
            ans_list = list()
            cfg_dir_path = os.path.dirname(os.path.abspath(cfg_path))
            with open(cfg_path, "r") as f:
                cfg_file = json.load(f)
                sub_cfgs_inc = cfg_file.pop("!include",list())
                sub_cfgs_excl = cfg_file.pop("!exclude",list())
                sub_cfgs_inc = [sub_cfgs_inc] if not isinstance(sub_cfgs_inc, list) else sub_cfgs_inc
                sub_cfgs_excl = [sub_cfgs_excl] if not isinstance(sub_cfgs_excl, list) else sub_cfgs_excl
                for sub_cfg_path in sub_cfgs_inc:                   
                    sub_cfg_fpath = os.path.abspath(os.path.join(cfg_dir_path, sub_cfg_path)) if not os.path.isabs(sub_cfg_path) else sub_cfg_path
                    d,exl = rec_load_cfgs(sub_cfg_fpath)
                    ans_dict[sub_cfg_fpath] = d
                    ans_list += exl
                for excl_fpath in sub_cfgs_excl:
                    excl_cfg_fpath = os.path.abspath(os.path.join(cfg_dir_path, excl_fpath)) if not os.path.isabs(excl_fpath) else excl_fpath           
                    ans_list.append(excl_cfg_fpath)
            return ans_dict, ans_list
        
        def rem_key_from_dict(cfg_dict: Dict, excl_key: str) -> Dict:
            if excl_key in cfg_dict:
                del cfg_dict[excl_key]
            new_dict = OrderedDict()
            for k in cfg_dict.keys():
                new_dict[k] = rem_key_from_dict(cfg_dict[k], excl_key)
            cfg_dict.update(new_dict)
            return cfg_dict
        
        def collect_dict_keys(cfg_dict: Dict) -> List:
            #ans = list(cfg_dict.keys())
            ans = list()
            for k,v in cfg_dict.items():
                ans += collect_dict_keys(v) + [k]
            return ans
            
        cfg_dict = LastUpdatedOrderedDict()
        cfg_excl_list = list()
        if not isinstance(cfg_paths, list):
            cfg_paths = [cfg_paths]
        for cfg_path in cfg_paths:
            cfg_fpath = os.path.abspath(cfg_path)
            d,l = rec_load_cfgs(cfg_fpath)
            cfg_dict[cfg_fpath] = d
            cfg_excl_list += l
            
        for excl_key in cfg_excl_list:
            cfg_dict = rem_key_from_dict(cfg_dict, excl_key)
            
        final_cfg_list = list(collect_dict_keys(cfg_dict))
        return final_cfg_list
        
                
    def load_params_from_cfg_file(self, cfg_path: str) -> Dict:
        """Load configuration from file(s).
        If JSON file has a field "!include" on a first level and it has a list (or single string) with name of file(s), than all of them will be loading at the beggining one by one.

        Args:
            cfg_path (str): path to the configuration file

        Returns:
            Dict: configuration
        """
        ans = dict()
        with open(cfg_path, 'r') as f:
            cfg_file = json.load(f)
            for k in ["!include", "!exclude"]:
                if k in cfg_file:
                    del cfg_file[k]
        return cfg_file


    def remove_params(self, params) -> Dict[str, None]:
        ans = {}
        for p in params:
            add_param = True
            for pn in self.args_notsec:
                if p == pn:
                    add_param = False
                    break
            if add_param:
                for ps in self.sec_parsers:
                    if p.startswith(f'{ps}.'):
                        add_param = False
                        break
            if add_param:
                ans[p] = params[p]
        return ans

    def load_params2attrs(self, **kwargs) -> None:
        p = self._convert_params_to_pyramid_dict(**kwargs)
        self._load_params_from_pyramid_dict(**p)

    def _convert_params_to_pyramid_dict(self, **params) -> Dict[str, None]:
        ans = {}
        for p, v in params.items():
            item_name_path = p.split('.')
            cur_arr = ans
            last_name = item_name_path[-1]
            for item_name in item_name_path[:-1]:
                if item_name not in cur_arr:
                    cur_arr[item_name] = {}
                cur_arr = cur_arr[item_name]
            cur_arr[last_name] = v
        return ans

    def _load_params_from_pyramid_dict(self, **params) -> None:
        for p in self.args_notsec:
            if p in params:
                self.args_notsec[p] = params[p]
        for p in self.sec_parsers:
            if p in params:
                self.sec_parsers[p]._load_params_from_pyramid_dict(**params[p])


if __name__ == '__main__':
    parser_ori = argparse.ArgumentParser()
    parser = ArgParserDecorator(parser_ori)

    parser.add_argument('-arg1')
    parser.add_argument('-arg2')
    sub_module1 = parser.add_sub_section_parser('module1')
    sub_module1.add_argument('arg1')
    sub_module1.add_argument('arg2')
    sub_module2 = parser.add_sub_section_parser('module2')
    sub_module2.add_argument('arg1')
    sub_module2.add_argument('arg2')

    args, unknown_args = parser.parse_known_args()
    sub_args = parser.get_params()
    print(f'args: {args}\nunknown_args: {unknown_args}\nsub_args: {sub_args}')
