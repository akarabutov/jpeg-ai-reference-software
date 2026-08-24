#!/bin/bash
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


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
work_dir=${SCRIPT_DIR}/../
DATA_ROOT=${work_dir}/data
cd ${work_dir}

export PYTHONPATH=${PYTHONPATH}:${work_dir}

python3 -m scripts.acc_train_scripts.acc_train_local \
       --data_dir ${DATA_ROOT}/jpegai_training_random_crop/ \
       --lst ${DATA_ROOT}/jpegai_training_random_crop/jpegai_training_set512_random_crop_16.txt \
       --val_data_dir ${DATA_ROOT}/jpegai_validation_set/ \
       --val_lst ${DATA_ROOT}/jpegai_validation_set/jpegai_validation_set_10.txt \
       --train_url ${work_dir}/train_results/ 
       
       ## Disable automatic testing
       # --use_automatic_testing 0 \

       ## Resume from pretrained models
       # --copy_to_train_url_dir models/VM_common/train_stages
       # --resume_from_stage MSE_VariableRate_12

       ## Freeze entropy part
       # --freeze_entropy_part 1

       ## Train only analysis part (encoder):
       # --train_only_analysis_part 1

       ## Train only synthesis part (decoder):
       # --train_only_synthesis_part 1

       ## Train only BOP:
       # --vae_encoder_type_list bop \
       # --vae_decoder_type_list bop \
       # --loss_weights 1 \
       # --cfg_path tools_off.json oper_point/bop.json

       ## Train only HOP:
       # --vae_encoder_type_list hop \
       # --vae_decoder_type_list hop \
       # --loss_weights 1 \
       # --cfg_path tools_off.json oper_point/hop.json

       ## Train encoder BOP, decoder SOP:
       # --vae_encoder_type_list bop \
       # --vae_decoder_type_list sop \
       # --loss_weights 1 \
       # --cfg_path tools_off.json oper_point/bopEnc_sopDec.json

       ## Train encoder <ENC>, decoder <DEC>:
       # --vae_encoder_type_list <ENC> \
       # --vae_decoder_type_list <DEC> \
       # --loss_weights 1 \
       # --cfg_path tools_off.json oper_point/common.json oper_point/<ENC>_Enc.json oper_point/<DEC>_Dec.json
