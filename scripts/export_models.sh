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


OUTPUT_DIR=results/models

python scripts/run_eval_script.py --cfg cfg/eval/onnx.json --module src.models_export.scripts.eval $OUTPUT_DIR -post_filters.tools ""
python -m src.models_export.scripts.eval --cfg ./cfg/tools_on.json ./cfg/oper_point/bop.json --out_dir $OUTPUT_DIR
#rm -R results/models/model/CCS_SGMM/tools_*/model_*/*.onnx
analyze_op=("bop" "hop")
encoder_ids=("0" "1")
comp_short=("y" "uv")
comp_short_up=("Y" "UV")
comp_names=("primary" "secondary")
synthesis_op=("sop" "bop" "hop")
synthesis_ids=("0" "1" "2")
syn_op_dirs=("bopEnc_sopDec" "bop" "hop")
betas=("0.002" "0.012" "0.075" "0.5")
tools_count=4

# Common part
op=${analyze_op[0]}
for (( tool_id=0; tool_id<$tools_count; tool_id++)); do
    for (( comp_id=0; comp_id<${#comp_short[@]}; comp_id++ )); do
        comp=${comp_short[$comp_id]}
        comp_out=${comp_names[$comp_id]}
        CUR_DIR=$OUTPUT_DIR/common/model_${tool_id}/${comp_out}
        mkdir -p $CUR_DIR

        cp -R $OUTPUT_DIR/${op}/tools_off-GPU/model/CCS_SGMM/tools_${tool_id}/model_${comp}/common_modules $CUR_DIR/
        cp -R $OUTPUT_DIR/${op}/tools_off-GPU/model/CCS_SGMM/tools_${tool_id}/model_${comp}/common_modules/quantizer/gain_unit_mlog.csv $CUR_DIR/
    done
done
# Copy unique Z distribution
cp models/VM_common_int/transition_table_z*.csv $OUTPUT_DIR/common/


# Encoders 
for (( op_idx=0; op_idx < ${#analyze_op[@]}; op_idx++ )); do
    op=${analyze_op[$op_idx]}
    op_id=${encoder_ids[$op_idx]}
    for (( tool_id=0; tool_id<$tools_count; tool_id++)); do
        for (( comp_id=0; comp_id<${#comp_short[@]}; comp_id++ )); do
            comp=${comp_short[$comp_id]}
            comp_out=${comp_names[$comp_id]}
            CUR_DIR=$OUTPUT_DIR/enc_dec/model_${tool_id}/${comp_out}
            mkdir -p $CUR_DIR

            cp $OUTPUT_DIR/${op}/tools_off-GPU/model/CCS_SGMM/tools_${tool_id}/model_${comp}/analysis.onnx ${CUR_DIR}/analysis_${op_id}.onnx
        done
    done
done

# Decoders
for (( op_idx=0; op_idx < ${#synthesis_op[@]}; op_idx++ )); do
    op_dirname=${syn_op_dirs[$op_idx]}
    op=${synthesis_op[$op_idx]}
    op_id=${synthesis_ids[$op_idx]}
    for tool_id in {0..3}; do
        for (( comp_id=0; comp_id<${#comp_short[@]}; comp_id++ )); do
            comp=${comp_short[$comp_id]}
            comp_up=${comp_short_up[$comp_id]}
            comp_out=${comp_names[$comp_id]}
            CUR_DIR=$OUTPUT_DIR/enc_dec/model_${tool_id}/${comp_out}
            mkdir -p $CUR_DIR

            cp $OUTPUT_DIR/${op_dirname}/tools_off-GPU/model/CCS_SGMM/tools_${tool_id}/model_${comp}/synthesis.onnx ${CUR_DIR}/synthesis_${op_id}.onnx
            cp models/VM_common_int/${comp_up}_${betas[$tool_id]}.csv ${CUR_DIR}/z_map_table.csv
        done
    done
done

mv $OUTPUT_DIR/post_filters $OUTPUT_DIR/ICCI
mv $OUTPUT_DIR/common/transition_table_z_* $OUTPUT_DIR/me-tANS/

# Delete folders with results
for dir_name in ${syn_op_dirs[@]}; do
    rm -R ${OUTPUT_DIR}/${dir_name}
done
rm -R ${OUTPUT_DIR}/model