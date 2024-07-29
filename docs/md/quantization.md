# Quantization of entropy
Search quantization parameters of each layer in hyper (scale) decoder.

Suggested calibration set is at ./data/calibration_set which is a subset of validation data.

# Running quantization of VM's models

Use command line:
```
./scripts/quantize_model.sh 
```

The script will quantize models from directories `models/VM_base` and `models/VM_high`, and store them to directories `models/VM_base_int` and `models/VM_high_int`.