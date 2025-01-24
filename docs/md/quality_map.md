# Quantization map

You can set quantization map based on a mask, which should be represented as PNG file. Mask area should be with white color. Put the file with mask to a directory `data/test_mask` with the same names as original files.

Run simulation of `<PROFILE>` by the following command:

```
python -m src.reco.scripts.eval --out_dir <OUTPUT> --coding_type enc_dec --cfg cfg/tools_off.json cfg/tools/quality_map.json ./cfg/profiles/<PROFILE>.json --use_qual_map 1 
```

