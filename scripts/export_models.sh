#!/bin/bash

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