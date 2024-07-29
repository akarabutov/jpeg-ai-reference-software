#!/bin/bash

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
       # --loss_weights 1,1
       # --cfg_path tools_off.json oper_point/bop.json

       ## Train only HOP:
       # --vae_encoder_type_list hop \
       # --vae_decoder_type_list hop \
       # --loss_weights 1,1
       # --cfg_path tools_off.json oper_point/hop.json

       ## Train encoder BOP, decoder SOP:
       # --vae_encoder_type_list bop \
       # --vae_decoder_type_list sop \
       # --loss_weights 1,1
       # --cfg_path tools_off.json oper_point/bopEnc_sopDec.json

       ## Train encoder <ENC>, decoder <DEC>:
       # --vae_encoder_type_list <ENC> \
       # --vae_decoder_type_list <DEC> \
       # --loss_weights 1,1
       # --cfg_path tools_off.json oper_point/common.json oper_point/<ENC>_Enc.json oper_point/<DEC>_Dec.json
