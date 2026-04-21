python -m src.reco.scripts.eval\
    --in_dir /ssd/20240213_eicci_training/JPEGAI_dataset\
    --out_dir ./results/recorder/bop\
    --cfg ./cfg/tools_off.json ./cfg/oper_point/bop.json\
    --coding_type enc\
    --record_for_eicci 1\
    --calc_encoder_metrics 0


python -m src.reco.scripts.eval\
    --in_dir /ssd/20240213_eicci_training/JPEGAI_dataset\
    --out_dir ./results/recorder/hop\
    --cfg ./cfg/tools_off.json ./cfg/oper_point/hop.json\
    --coding_type enc\
    --record_for_eicci 1\
    --calc_encoder_metrics 0
