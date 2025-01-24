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
from typing import List, Dict

class FileWrapper:
    def get_supported_ext(self) -> List[str]:
        return []
    
    def comment_text(self, txt: str) -> str:
        pass
    
    def start_processing_line(self) -> int:
        return 0
    
    def check_existance(self, file_txt: str, txt: str) -> bool:
        if self.start_processing_line() == 0:
            return file_txt.startswith(txt)
        else:
            lines = file_txt.split('\n')
            rest_text = "\n".join(lines[self.start_processing_line():])
            return rest_text.startswith(txt)
            
    
    def process(self, filename: str, lic_text: str, verbose: bool = False) -> None:
        lic_commented = self.comment_text(lic_text)
        with open(filename, "r") as f:
            file_txt = f.read()
        if not self.check_existance(file_txt, lic_commented):
            with open(filename, "w") as f:
                if self.start_processing_line() == 0:
                    f.write(lic_commented)
                    f.write('\n\n')
                    f.write(file_txt)
                else:
                    lines = file_txt.split('\n')
                    above_text = "\n".join(lines[:self.start_processing_line()])
                    rest_text = "\n".join(lines[self.start_processing_line():])
                    f.write(above_text)
                    f.write('\n')
                    f.write(lic_commented)
                    f.write('\n\n')
                    f.write(rest_text)
                    
            if verbose:
                print(f"File {filename} was processed")
        else:
            if verbose:
                print(f"! File {filename} already has the license")

class CppWrapper(FileWrapper):
    def get_supported_ext(self) -> List[str]:
        return ['cpp', 'h']
    
    def comment_text(self, txt: str) -> str:
        return f"/* {txt} */"    
    
class ShWrapper(FileWrapper):
    def get_supported_ext(self) -> List[str]:
        return ['sh']
    
    def start_processing_line(self) -> int:
        return 1

    def comment_text(self, txt: str) -> str:
        return "# " + txt.replace("\n", "\n# ")    
    
class PyWrapper(FileWrapper):
    def get_supported_ext(self) -> List[str]:
        return ['py']
    
    def comment_text(self, txt: str) -> str:
        return "# " + txt.replace("\n", "\n#")


class FileProcessing:

    LIC_TEXT=r"""The copyright in this software is being made available under the BSD
 License, included below. This software may be subject to other third party
 and contributor rights, including patent rights, and no such rights are
 granted under this license.

 Copyright (c) 2010-2022, ITU/ISO/IEC
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 THE POSSIBILITY OF SUCH DAMAGE."""
    
    def __init__(self):
        self.wrappers = list()
        self.wrappers.append(PyWrapper())
        self.wrappers.append(CppWrapper())
        self.wrappers.append(ShWrapper())
        
        self.all_supported_formats = list()
        for w in self.wrappers:
            self.all_supported_formats += w.get_supported_ext()
    
    def get_args(self) -> Dict:
        parser = argparse.ArgumentParser()
        parser.add_argument("root_dir", help=r"Root directory")
        parser.add_argument("--supported_formats", nargs="+", choices=self.all_supported_formats, help=r"Root directory")
        parser.add_argument('--verbose', default=False, action="store_true", help=r'Verbose mode')
        ans = vars(parser.parse_args())
        return ans

    def process(self, root_dir: str, supported_formats: List[str], verbose: bool = False) -> None:
        for dirpath, dnames, fnames in os.walk(root_dir):
            for f in fnames:
                ext = os.path.splitext(f)[1][1:]
                if ext in self.all_supported_formats:
                    full_path = os.path.join(dirpath, f)
                    for w in self.wrappers:
                        if ext in w.get_supported_ext():
                            w.process(full_path, self.LIC_TEXT, verbose)
                            
if __name__ == "__main__":
    processing = FileProcessing()
    
    args = processing.get_args()
    processing.process(**args)