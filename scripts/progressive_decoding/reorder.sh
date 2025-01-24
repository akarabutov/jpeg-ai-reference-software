script_dir=$(cd $(dirname $0); pwd)
# The copyright in this software is being made available under the BSD
#  License, included below. This software may be subject to other third party
#  and contributor rights, including patent rights, and no such rights are
#  granted under this license.
# 
#  Copyright (c) 2010-2022, ITU/ISO/IEC
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
# 
#  * Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
#  be used to endorse or promote products derived from this software without
#  specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
#  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
#  THE POSSIBILITY OF SUCH DAMAGE.

echo ${script_dir}
proj_root=`dirname $(dirname ${script_dir})`
model_root=${proj_root}/models
echo ${model_root}

# VM_bop
cd ${model_root}/VM_bop
python ${script_dir}/bop_decoder.py
python ${script_dir}/bop_encoder.py
cd -

# VM_sop
cd ${model_root}/VM_sop
python ${script_dir}/sop_decoder.py
cd -

# VM_hop
cd ${model_root}/VM_hop
python ${script_dir}/hop_decoder.py
python ${script_dir}/hop_encoder.py
cd -

# VM_common_int
cd ${model_root}/VM_common_int
python ${script_dir}/quantized_common.py
cd -

# this one must be the last one
cd ${model_root}/VM_common_int
python ${script_dir}/channel_wise_entropy.py
cd -
