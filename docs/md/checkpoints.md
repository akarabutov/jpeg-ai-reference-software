# Checkpoints processing

## Run evaluation with new checkpoints

Put checkpoints to directory `models/<MODEL_NAME>`, where `<MODEL_NAME>` is a name of the model which will be used in the test.

Set model name to parameter `ckpt_model_name` and list of checkpoints to `ckpt_files`. For example, to set the new model to CCS model through command line you need to add the following arguments:

    -model.CCS_SGMM.ckpt_model_name <MODEL_NAME> [-model.CCS_SGMM.ckpt_files <CHECKPOINT1>.pth [<CHECKPOINT2>.pth [<CHECKPOINT3>.pth [...]]]]

where `<CHECKPOINT`*N*`>.pth` is name of checkpoints in the directory `models/<MODEL_NAME>`.

In the case of usage script `src.reco.scripts.eval` you can add argument `--skip_models_check` to skip checking MD5 hash of the checkpoints.

## Update existing checkpoints

Put checkpoints to directory `models/<MODEL_NAME>`, where `<MODEL_NAME>` is a name of target model. You can execute only command:

    python scripts/process_model.py <MODEL_NAME>

or execute commands manually:

    dvc add models/<MODEL_NAME>/*.pth
    dvc commit models/<MODEL_NAME>/*.pth

## Push existing checkpoints to server

To push checkpoints of model `<MODEL_NAME>` from directory `models/<MODEL_NAME>` to sFTP server run the following command:

    python scripts/process_model.py <MODEL_NAME> --push

or execute commands manually:

    dvc add models/<MODEL_NAME>/*.pth
    dvc commit models/<MODEL_NAME>/*.dvc
    dvc push models/<MODEL_NAME>/*.dvc
    git add models/<MODEL_NAME>/*.dvc

## Pack existing checkpoints for uploading them on global sFTP

To pack checkpoints of model `<MODEL_NAME>` to archive run the following command:

    python scripts/process_model.py <MODEL_NAME> --pack

or execute commands manually:

    dvc add models/<MODEL_NAME>/*.pth
    dvc commit models/<MODEL_NAME>/*.dvc
    git add models/<MODEL_NAME>/*.dvc
    tar -cvzf <MODEL_NAME>.tgz models/<MODEL_NAME>/*.pth

Archive will be stored in root directory with name `<MODEL_NAME>.tgz`. If you place this archive to directory `/uploads` on global sFTP, it will automatically put checkpoints to global cache.
Request credentials from SW coordinators to get write access to the directory.
