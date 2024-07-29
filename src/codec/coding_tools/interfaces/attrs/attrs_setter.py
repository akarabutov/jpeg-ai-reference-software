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

from contextlib import ContextDecorator

from ..base import BaseModule


class ToolsAttributeSetter(ContextDecorator):
    def __init__(self, tools, attributes: dict):
        self.attributes = attributes
        if not isinstance(tools, list):
            tools = [tools]
        self.tools = tools
        self.stored_attributes = []

    def __enter__(self):

        for tool in self.tools:
            if tool is not None:
                old_attrs = {}
                for attr, val in self.attributes.items():
                    old_attrs[attr] = getattr(tool, attr, None)
                    setattr(tool, attr, val)
                self.stored_attributes.append(old_attrs)

        return self

    def __exit__(self, *exc):
        for (tool, attrs) in zip(self.tools, self.stored_attributes):
            for attr, val in attrs.items():
                setattr(tool, attr, val)
        return False


class AttributeSetter(ContextDecorator):
    def __init__(self, root_module, modules_type, attributes: dict):
        self.attributes = attributes
        if not isinstance(modules_type, list):
            modules_type = [modules_type]
        self.modules_type = modules_type
        self.root_module = root_module

    @staticmethod
    def set_attrs_value(module: BaseModule, modules_type: list, new_attrs: dict) -> None:
        for mt in modules_type:
            if isinstance(module, mt):
                for n, v in new_attrs.items():
                    old_v = getattr(module, n, None)
                    setattr(module, n, v)
                    setattr(module, f'old_value_{n}', old_v)
                return

    @staticmethod
    def revert_attrs_value(module: BaseModule, modules_type: list, new_attrs: dict) -> None:
        for mt in modules_type:
            if isinstance(module, mt):
                for n, v in new_attrs.items():
                    old_v = getattr(module, f'old_value_{n}', None)
                    setattr(module, n, old_v)
                return

    def __enter__(self):
        self.root_module.for_each_child(
            lambda _, m: AttributeSetter.set_attrs_value(m, self.modules_type, self.attributes))
        return self

    def __exit__(self, *exc):
        self.root_module.for_each_child(
            lambda _, m: AttributeSetter.revert_attrs_value(m, self.modules_type, self.attributes))
        return False
