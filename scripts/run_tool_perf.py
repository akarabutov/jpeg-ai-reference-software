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
from typing import Dict, List
from tempfile import NamedTemporaryFile
from run_eval_script import run_eval_script
from merge_summaries import process_summaries


def get_args():
    cur_file_path=os.path.dirname(os.path.abspath(__file__))
    cfg_base_path=os.path.abspath(os.path.join(cur_file_path, os.pardir, "cfg", "eval", "base.json"))
    parser = argparse.ArgumentParser()
    parser.add_argument('TOOL_CFG_DIR', type=str, help="Path to a directory with configuration files")
    parser.add_argument('OUTPUT_BASE_DIR', type=str, help="Output base directory")
    parser.add_argument('--module', '-m', type=str, default="src.reco.scripts.eval", help="Module for simulation")
    parser.add_argument('--base-cfg', type=str, default=cfg_base_path, help="Path to the base configuration file")
    return parser.parse_known_args()


def process_tool_tests(OUTPUT_BASE_DIR: str, TOOL_CFG_DIR: str, base_cfg: str, module: str = None, additional_args: List[str] = None) -> None:
    base_cfg = os.path.abspath(base_cfg)
    cfg = {
        "!include": [base_cfg],
        "configurations": {
        }        
    }
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    for cfg_path in os.listdir(TOOL_CFG_DIR):
        if cfg_path.endswith(".json"):
            tool_file_name = os.path.basename(cfg_path)
            tool_name = os.path.splitext(tool_file_name)[0]
            cfg['configurations'][tool_name] = {
               "--cfg": [os.path.abspath(os.path.join(TOOL_CFG_DIR, cfg_path))] 
            }
    with NamedTemporaryFile("w") as f:
        json.dump(cfg, f)
        f.flush()
        run_eval_script(OUTPUT_BASE_DIR, f.name, module, additional_args)
    for op_dir in os.listdir(OUTPUT_BASE_DIR):
        full_path = os.path.abspath(os.path.join(OUTPUT_BASE_DIR, op_dir))
        if os.path.isdir(full_path):
            process_summaries(full_path, fn_prefix=op_dir, anchor="anchor")
        
    
def main():
    args, additional_args = get_args()
    process_tool_tests(**vars(args), additional_args=additional_args)
        
if __name__ == "__main__":
    main()