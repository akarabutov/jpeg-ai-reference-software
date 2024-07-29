# Training Accelerated VM

## Setup Environment
The enviroment is the same as for running inference/test. See [Readme](../../../../README.md).
## Prepare Dataset

Since the images in JPEGAI training dataset are pretty big, which makes the loading of data in
training phase very slow, the dataset is randomly cropped. Please use random crop example from `scripts/crop_image.sh` to reproduce cropped version JPEG AI training dataset.

## Train
The training process involves 4 models train by 4 serial stages.
Bash scripts which may be used for train reproduction are placed `scripts/acc_train_scripts` folder. You may find instruction for starting training in [readme](../../../../scripts/acc_train_scripts/Readme.md).
