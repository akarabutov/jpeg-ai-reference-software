# eICCI training code

## 0. Note

Developed based on the **mmsr** github repository for super-resolution <https://github.com/andreas128/mmsr>

**mmsr** itself is licensed under Apache 2.0 <https://github.com/andreas128/mmsr/blob/master/LICENSE>

## 1. Creating the environemnt

```
conda create --name eicci python=3.6.7
activate eicci
```

### Option A - requirements

```
pip install -r eicci_requirements.txt
```

### Option B - manual

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install lmdb pyyaml scipy tensorboard pytorch-msssim
pip install --upgrade git+<https://github.com/tensorpack/dataflow.git>
pip install opencv-python --verbose
```

*note: opencv-python installation can take approx 1 hour*

## 2. Training

### 2.1 Run the recorder

Set the input directory `--in_dir` for recordings to the JPEG-AI training dataset in `run_recorder.sh`, and run

```
./run_recorder.sh
```

### 2.2 Create dataset

Creating the dataset has the following steps: I. move the recorded files in separate folders, one for each qp; II. extract subimages (patches) of each image; III. group the subimages into lmdb databases (for fast access)

I. Copy the files:

```
mv ...results/recorder/bop/rec/*_012.yuv dataset/yuv/bop_012
mv ...results/recorder/bop/rec/*_025.yuv dataset/yuv/bop_025
mv ...results/recorder/bop/rec/*_050.yuv dataset/yuv/bop_050
mv ...results/recorder/bop/rec/*_075.yuv dataset/yuv/bop_075
mv ...results/recorder/bop/rec/*_100.yuv dataset/yuv/bop_100
mv ...results/recorder/bop/ori/*.yuv dataset/yuv/bop_org

mv ...results/recorder/hop/rec/*_012.yuv dataset/yuv/hop_012
mv ...results/recorder/hop/rec/*_025.yuv dataset/yuv/hop_025
mv ...results/recorder/hop/rec/*_050.yuv dataset/yuv/hop_050
mv ...results/recorder/hop/rec/*_075.yuv dataset/yuv/hop_075
mv ...results/recorder/hop/rec/*_100.yuv dataset/yuv/hop_100
mv ...results/recorder/bop/ori/*.yuv dataset/yuv/hop_org
```

*note: hop_org and bop_org most likely have identical contents. We keep separate folders in case one wants to use different subsets. Feel free to use only one folder and change scripts accordingly.

II/III. extract subimages and create lmdb database

```
cd codes/data_scripts/
python extract_subimages_eicci_bop.py
python create_lmdb_eicci_bop.py
python extract_subimages_eicci_hop.py
python create_lmdb_eicci_hop.py
```

*note: All input and output paths are defined at the top of each script. extract_subimages takes input from .../dataset/val and output is in .../dataset/subimages. create_lmdb takes input from .../dataset_subimages and output is in .../dataset/lmdb*

### 2.3 Validation dataset

Validation dataset is used only for tracking the training progress with tensorflow. The validation is not used for selection of the filter (we take the last filter from iteration 300000). One can use separate validation set, or one can use the trainin set for validation - as long as tensorboard output is not used for selection of the filter, there is no overfitting.

You need to have the same amount of files in each .../dataset/val/... subfolder. Filenames do not matter, as long as they match when sorted.

Here we use 1 file for validation. Feel free to include more files in each folder. The validation dataset does not influence the training, only the tensorboard output.

```
cp ...results/recorder/bop/rec/VM_02133_TR_960x540_8bit_444_012.yuv dataset/val/bop_012
cp ...results/recorder/bop/rec/VM_02133_TR_960x540_8bit_444_025.yuv dataset/val/bop_025
cp ...results/recorder/bop/rec/VM_02133_TR_960x540_8bit_444_050.yuv dataset/val/bop_050
cp ...results/recorder/bop/rec/VM_02133_TR_960x540_8bit_444_075.yuv dataset/val/bop_075
cp ...results/recorder/bop/rec/VM_02133_TR_960x540_8bit_444_100.yuv dataset/val/bop_100
cp ...results/recorder/bop/ori/02133_TR_960x540_8bit_444.yuv dataset/val/bop_org

cp ...results/recorder/hop/rec/VM_02133_TR_960x540_8bit_444_012.yuv dataset/val/hop_012
cp ...results/recorder/hop/rec/VM_02133_TR_960x540_8bit_444_025.yuv dataset/val/hop_025
cp ...results/recorder/hop/rec/VM_02133_TR_960x540_8bit_444_050.yuv dataset/val/hop_050
cp ...results/recorder/hop/rec/VM_02133_TR_960x540_8bit_444_075.yuv dataset/val/hop_075
cp ...results/recorder/hop/rec/VM_02133_TR_960x540_8bit_444_100.yuv dataset/val/hop_100
cp ...results/recorder/hop/ori/02133_TR_960x540_8bit_444.yuv dataset/val/hop_org
```

### 2.4 Training

Training uses a yml configuration file which is in ...codes/options/train. All needed yml files are provided.

Training needs <4GB VRAM, we are able to run 5 training sessions in parallel on a GPU with 24GB ram. On machine with 2x TitanX GPUs we train 10 filters in parallel. As long as the lmdb databses are on a fast SSD drive, file I/O is not a bottleneck.

The training saves state each 2000 iterations, which allows restarting training from a saved point. If you need to restart a training, check the .../codes/options/train/*.yml files, where is an example how to provide the path to a saved model and saved state.

The full training runs for 300000 iterations, and we take the last saved model. On a TitanX GPU training of 5 filters in parallel takes ~30 hours. We training all 20 filters (5qps, mse/msssim, hop/bop) for 3-4 days.

In the ...codes folder there is a shell script which helps selecting training configuration. Depending on the GPU, you an run one or more of those in parallel.

```
./train_eicci.sh -p bop -l mse -q 012
./train_eicci.sh -p bop -l mse -q 025
./train_eicci.sh -p bop -l mse -q 050
./train_eicci.sh -p bop -l mse -q 075
./train_eicci.sh -p bop -l mse -q 100

./train_eicci.sh -p bop -l mss -q 012
./train_eicci.sh -p bop -l mss -q 025
./train_eicci.sh -p bop -l mss -q 050
./train_eicci.sh -p bop -l mss -q 075
./train_eicci.sh -p bop -l mss -q 100

./train_eicci.sh -p hop -l mse -q 012
./train_eicci.sh -p hop -l mse -q 025
./train_eicci.sh -p hop -l mse -q 050
./train_eicci.sh -p hop -l mse -q 075
./train_eicci.sh -p hop -l mse -q 100

./train_eicci.sh -p hop -l mss -q 012
./train_eicci.sh -p hop -l mss -q 025
./train_eicci.sh -p hop -l mss -q 050
./train_eicci.sh -p hop -l mss -q 075
./train_eicci.sh -p hop -l mss -q 100
```

### 3. Results

Results will be in .../experiments/ in a folder with the name of the experiment. The ame of the experiment is defined at the top of the yml file, and uses the format **eicci\_{bop/hop}\_{mse/mss}\_{012/025/050/075/100}**. Training runs for 300k iterations, and saves a model each 1000 iteration. The ouput is the last saved model, e.g.

```
.../experiments/eicci_bop_mse_025/models/latest_G.pth
```

(or, 300000_G.pth, which is the same)
