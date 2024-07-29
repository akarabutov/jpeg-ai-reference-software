script_dir=$(cd $(dirname $0); pwd)
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
