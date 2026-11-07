# Training Accelerated VM

## Setup Environment
The enviroment is the same as for running inference/test. See [Readme](../../../../README.md).
## Prepare Dataset

Since the images in JPEGAI training dataset are pretty big, which makes the loading of data in
training phase very slow, the dataset is randomly cropped. Please use random crop example from `scripts/crop_image.sh` to reproduce cropped version JPEG AI training dataset.

## Train
The training process involves 4 models train by 4 serial stages.
Bash scripts which may be used for train reproduction are placed `scripts/acc_train_scripts` folder. You may find instruction for starting training in [readme](../../../../scripts/acc_train_scripts/Readme.md).

## Implementation

A stage is trained by `multistages_train/train.py` with [PyTorch Lightning](https://lightning.ai/docs/pytorch/1.9.5/)
(`pytorch-lightning==1.9.5`, the newest release supporting the pinned `torch==1.10.2`; Lightning 2.x works too
with a newer torch). The Lightning parts are in [lightning](lightning):

- `module.py` — `CCSLightningModule`: training step with manual optimization (AMP with the fp32 fallback,
  per-sub-network gradient clipping, random beta per step) and validation;
- `datamodule.py` — `CCSDataModule`: datasets, loaders and distributed samplers;
- `strategy.py` — `PerSubmoduleDDPStrategy`: DDP strategy keeping the per-sub-network DDP of `model/net.py`;
- `callbacks.py` — `StageOutputs` (validation table, `val_results.json`, TensorBoard, `<epoch>.pth`, `best.pth`,
  `last.ckpt`) and `AutomaticTesting`.

The outputs (`<epoch>.pth`, `best.pth`, `val_results.json`, TensorBoard logs) have the same format as before.
One stage can be run without the launcher, e.g. on one GPU:

```
python -m src.train.CCS.acc_train.multistages_train.train --data_dir <DIR> --lst <LIST> \
       --val_data_dir <DIR> --val_lst <LIST> --train_url <OUT>/<STAGE>/<BETA> --beta_list 0.002 ...
```

or on several GPUs with `python -m torch.distributed.run --nproc_per_node=<N> -m src.train.CCS.acc_train.multistages_train.train ...`.
See [docs/architecture/13-training.md](../../../../docs/architecture/13-training.md) for details.
