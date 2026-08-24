# Description of training launch scripts
* Training scrips create a separate process for each model
* Each process will go through all stages necessary for the training of the model
* If trainig sript crashes it is resumed from the last epoch of the same stage
* Automatic testing is done using the inference code every 4 epochs

## acc_train_local.py

### main
* creates a pool of workers
* each worker is given one beta to start training
* periodically main function checks if all workers are done and copies output to output directory

### run_stages_for_one_beta
The function called for each worker process.
* sets up parametes for all stages. depending on arguments and beta
* selects GPUs which should be used for current worker
* launches DDP multistage training for the current beta
* handles resuming from checkpoints
  * best of previous stage for a newly started stage
  * latest epoch if train crashed. if train was not successful (returncode of process not 0) will restart and resume from last epoch

# Instruction

1. Run a command `make download_train_ds`. It downloads training and validation datasets.
2. Training script uses two configuratioin files:
  * `cfg/train.json` with parameters of betas for training and sequence of training stages.
  * `cfg/train_satges.json` with description of training stages.

## Parameters of training
In the file [cfg/train.json](cfg/train.json) you can set the following parameters:

1. GPUs for training betas in a field `beta_2_gpus`. It has the following format:
```
     "beta_2_gpus": {
         "0.002": "GPU_ID, GPU_ID ",
         "0.007": " GPU_ID, GPU_ID ",
         "0.015": " GPU_ID, GPU_ID ",
         "0.05": " GPU_ID, GPU_ID ",
     }
```
  where the key is the target beta and the value is a comma-separaterd list of GPUs where training should be performed.

2. List of betas for training of variable rate control is set in a field `beta_2_betaList`:
```
    "beta_2_betaList": {
        "0.002": "0.0002,0.0005,0.001,0.002,0.004",
        "0.012": "0.007,0.01,0.012,0.015",
        "0.075": "0.03,0.05,0.075,0.1",
        "0.5": "0.1,0.2,0.5,0.75"
    }
```

3. Setting a list of training stages which will be performed sequentially:
```
    "stages": [
        "MSE_FixedRate_64",
        "Mixed_FixedRate_36",
        "Mixed_FixedRate_OnlyDec_20",
        "MSE_VariableRate_12", 
        "Data_Collection"
    ]
```
  tasks description one can find in the file [cfg/train_satges.json](cfg/train_satges.json).

The file is in JSON format with support of comments (see a package [commentjson](https://pypi.org/project/commentjson/)).

## Definition of training stages
Training stages are defined in the file [cfg/train_satges.json](cfg/train_satges.json). Format of defining of the stages:
```
{
  "STAGE_NAME": [
    ARG1,
    ARG2,
    ...
  ],
  ...
}
```
  where `STAGE_NAME` is a name of a stage, `ARG1`, `ARG2` and etc. are arguments of a command line of a script [train.py](src/train/CCS/acc_train/multistages_train/train.py).
  The file is in JSON format with support of usage of comments (see a package [commentjson](https://pypi.org/project/commentjson/)) and variables from training script.  
  We use a [script](src/codec/utils/templite.py) for adding variables' value. 

## Start training

Run the following command to start training:
```
make train
```
It will store all data to a directory `train_results`.

## Results of training

The output directory will contain:
      -	Logs from training
        /log/MSE_FixedRate_64
        /log/Mixed_FixedRate_36
        /log/Mixed_FixedRate_OnlyDec_20
        /log/MSE_VariableRate_12

      -	models for each epoch stores to `/<STAGE_NAME>/<BETA>/`,  where `<STAGE_NAME>` is a stage name (see teh section "Definition of training stages"); and `<BETA>` is training beta (see `beta_2_gpus` in the section "Parameters of training").

      -	if testing during training was enabled, test results for interim epochs stores to `/<STAGE_NAME>/<BETA>/<EP>_test/`, where `<EP>` is a number of test, i.e. 0,4,8 and etc.

      -	if testing during training was enabled, after all models are trained one can find summary.txt for 5 rate points (JPEG AI CTTC)  in `/<STAGE_NAME>/results/epoch<EP>/`, where `<EP>` is a number of test, i.e. 0,4,8 and etc.
