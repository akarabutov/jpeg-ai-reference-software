# JPEG-AI Reference software

This software package is the reference software for Rec. ITU-T T.840.1 | ISO/IEC 6048-1 JPEG AI learning-based image coding system (JPEG-AI). The reference software includes both encoder and decoder functionality.
Reference software is useful in aiding users of a image coding standard to establish and test conformance and interoperability, and to educate users and demonstrate the capabilities of the standard. For these purposes, this software is provided as an aid for the study and implementation of JPEG-AI.
The software has been jointly developed by the ITU-T Video Coding Experts Group (VCEG, Question 6 of ITU-T Study Group 16) and Joint Technical Committee ISO/IEC JTC 1, Information technology, Subcommittee SC 29, Coding of audio, picture, multimedia and hypermedia information.
A software manual, which contains usage instructions, can be found in the "docs" subdirectory of this software package.
The source code is stored in a Git repository. The most recent version can be retrieved using the following commands:

```
git clone https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git
cd jpeg-ai-reference-software
```

## System requirments
1. Ubuntu Linux 18.04 or later
2. CUDA 10.2+ or CUDA 11.3+.
2. List of packages (you may run `make setup_system` to install them):
    - doxygen 1.8.13
    - graphviz 2.40.1
    - git-lfs 3.0.2

## Setup Environment

1. Install reuirments:
    - On Ubuntu PC.
        Install [miniconda](https://docs.anaconda.com/miniconda/) and setup an environment by a command: `make configure`.
    
    - Docker container.
        To get Docker container run a command: `make run_docker`.

2. Build C++ libraries for testing: `make build_test_libs`.

## Downloading datasets and models

### Dataset for the reconstruction task

By default a dataset for evaluation locates in a directory `data/test`. 

Run a command `make download_test_ds` to download JPEG-AI evaluation dataset.

### Dataset for training

Training and Validation datasets could be downloaded by a command: `make download_test_ds`.

The training dataset will be stored to `data/jpegai_training_random_crop` and the validation dataset will be stored to `data/jpegai_validation_set`.

### Models

Run the following command for downloading models:
```
make download_models
```

## Evaluation of the reconstruction task

Evaluation over all images in the dataset:

```
make test
```
the results will be stored to a directory `results/test`.
The script automatically download models and checks there MD5 hashs.

Use the following command line to encode an image:

```
python -m src.reco.coders.encoder <IMAGE_PATH> <OUTPUT_STREAM_PATH> [--set_target_bpp <TARGET_BPPm100>] [--cfg <CFG1> [<CFG2> [<CFG3> ...]]]
```

where `<IMAGE_PATH>` is a path to the input image in PNG format, `<OUTPUT_STREAM_PATH>` is a path to the output bitstream, `<TARGET_BPPm100>` is a target bit per pixel multiplied by 100. Specify a list of the configuration files of the encoding. Configuration files load one by one. In a case of running tests without any tool, the command line is: `--cfg cfg/tools_off.json cfg/profiles/<PROFILE>.json`, where `<PROFILE>` is `simple`, `main` or `high`. In a case of running tests without all tools, the command line is: `--cfg cfg/tools_on.json cfg/profiles/<PROFILE>.json`. To run test with enabling only particular tools, use the following command line: `--cfg cfg/tools_off.json cfg/tools/<TOOL1>.json [cfg/tools/<TOOL2>.json ...] cfg/profiles/<PROFILE>.json`. Where `<TOOLN>.json` is one of the files from cfg/tools directory.


Run the following command to decode the bitstream file:

```
python -m src.reco.coders.decoder <INPUT_STREAM_PATH> <OUTPUT_PNG_IMAGE_PATH> 
```

where `<INPUT_STREAM_PATH>` is the path to the bitstream, `<OUTPUT_PNG_IMAGE_PATH>` is the path to the output PNG file.


## Documentation

You may find slides with SW design [here](docs/ppt/VM.pptx).



An example of a command line for training you can find in a file `scripts/train.sh`.
Additional information about setting parameters of training you can find [here](src/train/README.md).

### Quantization of trained models

Description of quantization process you can find in a [file](docs/md/quantization.md).

## Progressive decoding
To enable the progressive decoding functionality, please run `bash scripts/progressive_decoding/reorder.sh` to make the latent tensor be arranged in decerasing entropy order across the channel dimension.


### Checkpoints

You may find information about checkpoints processing [here](docs/md/checkpoints.md)



## List of 'make' commands

- `make setup_system` installs all necessary packages on your Ubuntu Linux.
- `make setup_env` creates conda environment (`jpeg_ai_vm`) install all necessary python's packages and build all necessary c++ libraries.
- `make build_test_libs` builds all necessary for test C++ libraries.
- `make build_train_libs` builds all necessary for training C++ libraries.
- `make build_libs` builds all C++ libraries for test and training.
- `make download_dvc_cache` downloads DVC cache from JPEG-AI's sFTP.
- `make download_test_ds` pulls test dataset from DVC cache.
- `make download_models` pulls models from DVC cache.
- `make download_train_ds` downloads training and validation datasets.
- `make test` runs test with the default configuration and store results to a directory `results/test`.
- `make unittest` runs unit tests.
- `make tool_ena` runs tools-off tests with only one tool enabled.
- `make tool_dis` runs tools-on tests with only one tool disabled.
- `make tool_perf` runs test `tool_ena` and `tool_dis`.
- `make train` runs training.
- `make export_models` exports models to ONNX and CSV files.
- `make run_docker` runs docker container.


## Troubleshooting

### Cannot download checkpoints from sFTP

First of all, check that you can connect to sFTP (step 5 in Set-up section). In the case of success try to run a test again.

Otherwise you may use http mirror for this and download checkpoints from global sFTP
manually by running a command: `make download_dvc_cache`
