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
    - p7zip-full (to unpack the training datasets)

## Setup Environment

1. Install reuirments:
    - On Ubuntu PC.
        Install [miniconda](https://docs.anaconda.com/miniconda/) and setup an environment by a command: `make configure`.
    
    - Docker container.
        To get Docker container run a command: `make run_docker`.

2. Build C++ libraries for testing: `make build_test_libs`.

## Evaluation of the reconstruction task

Evaluation over all images in the dataset:

```
activate jpeg_ai_vm
make test
```
the results will be stored to a directory `results/test`.
The script automatically download models and checks there MD5 hashs.

Use the following command line to encode an image:

```
activate jpeg_ai_vm
python -m src.reco.coders.encoder <IMAGE_PATH> <OUTPUT_STREAM_PATH> [--set_target_bpp <TARGET_BPPm100>] [--cfg <CFG1> [<CFG2> [<CFG3> ...]]]
```

where `<IMAGE_PATH>` is a path to the input image in PNG format, `<OUTPUT_STREAM_PATH>` is a path to the output bitstream, `<TARGET_BPPm100>` is a target bit per pixel multiplied by 100. Specify a list of the configuration files of the encoding. Configuration files load one by one. In a case of running tests without any tool, the command line is: `--cfg cfg/tools_off.json cfg/profiles/<PROFILE>.json`, where `<PROFILE>` is `simple`, `main` or `high`. In a case of running tests without all tools, the command line is: `--cfg cfg/tools_on.json cfg/profiles/<PROFILE>.json`. To run test with enabling only particular tools, use the following command line: `--cfg cfg/tools_off.json cfg/tools/<TOOL1>.json [cfg/tools/<TOOL2>.json ...] cfg/profiles/<PROFILE>.json`. Where `<TOOLN>.json` is one of the files from cfg/tools directory.


Run the following command to decode the bitstream file:

```
activate jpeg_ai_vm
python -m src.reco.coders.decoder <INPUT_STREAM_PATH> <OUTPUT_PNG_IMAGE_PATH> 
```

where `<INPUT_STREAM_PATH>` is the path to the bitstream, `<OUTPUT_PNG_IMAGE_PATH>` is the path to the output PNG file.


## Documentation

Detailed architecture documentation — repository layout, the engine/tool framework, the encoding
and decoding pipelines, the bitstream format, entropy coding, the neural networks, a per-tool
reference and the command-line tools, all with diagrams — is in
[docs/architecture](docs/architecture/README.md).

To read it as HTML, run `make docs` for the Doxygen site (`docs/html/index.html`, architecture
pages alongside the API reference) or `make docs_single` for one self-contained page
(`docs/architecture.html`). See [docs/doxygen/README.md](docs/doxygen/README.md) for details.

You may find slides with SW design [here](docs/ppt/VM.pptx).



## Training dataset

The training and validation datasets are published by ISO and by ITU. Run

```
make download_train_ds
```

and the script asks what is needed, showing how much each choice downloads:

1. which mirror to use — ISO (`standards.iso.org`) or ITU (`www.itu.int`);
2. which datasets — training, validation or both;
3. for the training set, whether the natural content is wanted as full-size images
   (~54 GiB) or as cropped patches (~164 GiB in four ranges), and which of the extra
   datasets — `SCC7000P2`, `HF2000`, `LQ7000`, `MD300`, `EXCEL300`, `PHF200`, `PHFA500`,
   `CP50`, ~5.5 GiB together — to add;
4. for the validation set, which form (cropped or full-size, ~3.8 GiB each) and a
   confirmation of the size;

then whether to unpack the archives and whether to keep them afterwards. Anything already on
disk is checked first: the script reports which archives are complete, which are short and by
how much, and which have already been unpacked, and offers to finish the incomplete ones, to
download everything again, or to leave what is there alone — each option with the number of
bytes it will actually transfer. It ends with a summary of the total download, the disk space
needed and the free space at the destination.

The catalogue is read from the mirror itself — the published directory index is walked and the
sizes are taken from it — so nothing is hard-coded and a renamed or added archive is picked up
automatically. `--list-remote` prints what a mirror offers and how each archive was classified;
`--check` compares all of it against what is already on disk without downloading anything.
Downloads resume, and the large datasets are published as split 7-Zip bundles
(`...bundle.7z.001`, `.002`, …), which are downloaded volume by volume and unpacked with `7z`
(`sudo apt install p7zip-full`, also installed by `make setup_system`).

Every answer also has an option, which makes the same run repeatable without questions:

| Option | Purpose |
| --- | --- |
| `--source {iso,itu}` | Mirror to download from |
| `--datasets {train,validation,both}` | Which datasets are needed |
| `--natural {full,patches,none}` | Full-size natural images or cropped patches |
| `--validation {cropped,full,all,none}` | Which form of the validation set |
| `--extras all\|none\|NAME,NAME` | Extra training datasets |
| `--existing {resume,redownload,skip}` | What to do with archives already on disk |
| `--reunpack` / `--no-reunpack` | Unpack again what is already unpacked |
| `--unpack` / `--no-unpack`, `--remove-archives` | What happens after the download |
| `--data-dir DIR`, `--archives-dir DIR` | Where the datasets and the archives go |
| `--save-answers FILE`, `--answers FILE` | Save the answers and replay them later |
| `--status`, `--check`, `--list-remote`, `--dry-run` | Inspect without downloading |
| `--yes` | Skip the final confirmation |

The result is the layout the training scripts expect:

```
data/jpegai_training/                                               full-size natural images
data/jpegai_training_random_crop/                                   training patches, with each
                                                                    extra dataset in its own
                                                                    subdirectory
data/jpegai_training_random_crop/jpegai_training_set512_random_crop_16.txt
data/jpegai_validation_set/                                         validation images
data/jpegai_validation_set/jpegai_validation_set_10.txt
```

What each archive produced is recorded in `data/.jpegai_datasets.json`, which is what lets a
later run tell downloaded from downloaded-and-unpacked.

An example of a command line for training you can find in a file `scripts/train.sh`.
Additional information about setting parameters of training you can find [here](src/train/README.md).


## List of 'make' commands

- `make setup_system` installs all necessary packages on your Ubuntu Linux.
- `make setup_env` creates conda environment (`jpeg_ai_vm`) install all necessary python's packages and build all necessary c++ libraries.
- `make build_test_libs` builds all necessary for test C++ libraries.
- `make download_train_ds` downloads and unpacks the training and validation datasets (see [Training dataset](#training-dataset)).
- `make test` runs test with the default configuration and store results to a directory `results/test`.
- `make unittest` runs unit tests.
- `make tool_ena` runs tools-off tests with only one tool enabled.
- `make tool_dis` runs tools-on tests with only one tool disabled.
- `make tool_perf` runs test `tool_ena` and `tool_dis`.
- `make export_models` exports models to ONNX and CSV files.
- `make run_docker` runs docker container.
- `make docs` builds the HTML documentation (API reference plus the architecture pages) to `docs/html`.
- `make docs_single` builds the architecture documentation as one self-contained page `docs/architecture.html`.
