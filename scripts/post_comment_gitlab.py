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

import argparse
import subprocess
import json
from typing import Dict
import gitlab
import os
import glob

def get_args(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("ACCESS_TOKEN", type=str, help="Access token") 
    parser.add_argument("PROJECT_ID", type=int, help="ID of the project")
    parser.add_argument("MERGE_REQUEST_ID", type=int, help="ID of MR")
    parser.add_argument("--url", type=str, default="https://jpeg-git.lx.it.pt", help="URL of the base project like https://jpeg-git.lx.it.pt")
    parser.add_argument("--files", type=str, default=[], nargs="+", help="List of pathes to files for uploading")
    parser.add_argument("--msg", type=str, default="", help="Text message")
    parser.add_argument("--only_if_files_exist", default=False, action="store_true", help="Post the message only if at least one of the files exist")
    
    return parser.parse_args(args)

def main():
    args = get_args()
    files_list = list()
    for fn in args.files:
        a = glob.glob(fn)
        files_list += a
        
    if args.only_if_files_exist:
        file_exist = False
        for fp in files_list:
            if os.path.exists(os.path.abspath(fp)):
                file_exist = True
        if not file_exist:
            return 

    gl = gitlab.Gitlab(args.url, private_token=args.ACCESS_TOKEN, ssl_verify=False)
    gl.auth()
    
    msg = args.msg
    project = gl.projects.get(args.PROJECT_ID)
    for i, fn in enumerate(files_list):
        f_data = project.upload(filename=os.path.basename(fn), filepath=os.path.abspath(fn))
        #f_data = upload_file(args.BASE_PROJECT_URL, fn, args.ACCESS_TOKEN)
        if i == 0:
            msg += "\n\n"
        msg += f"- {f_data['markdown']}\n"
    #post_message(args.BASE_PROJECT_URL, args.MERGE_REQUEST_ID, args.ACCESS_TOKEN, msg)
    mr = project.mergerequests.get(args.MERGE_REQUEST_ID)
    mr.discussions.create({'body': msg})
        
if __name__== "__main__":
    main()       