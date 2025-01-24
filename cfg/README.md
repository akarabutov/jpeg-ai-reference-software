# Directory with configuration files

The directory contains several kinds of configuration files:
1. Configuration of the current release.
2. Files with configuration of tools and operation points.
3. Configuration files for evaluation tool.


## Configuration of the current release

File [info.json](info.json) has the following information:
- the current version of the software (field `version`)
- the default configuration for a test (field `config`)

File [launch.json](launch.json) contains configuration for running tests and training in [Visual Studio Code](https://code.visualstudio.com/).

## Files with configuration of tools and operation points

### Configuration files of tools

Configuration of each tool locates in a directory [tools](tools). Each file in the directory has configuration only for a one tool.

The current directory contains the following configuration files for tools:
- [pipeline.json](pipeline.json) is a configuration file with pipeline of the codec. It should have only configuration of decoder. All encoder-wise configuration should be in a tool-specific configuration file in [tools](tools) directory.
- [CTC.json](CTC.json) is a base configuration file, which has default configuration for all operation points and tools.
- [tools_off.json](tools_off.json) is a configuration file for a case when all tools are disabled.
- [tools_on.json](tools_on.json) is a configuration file for a case when all tools are enabled.

It also contains the following directories:
- [tool_ena](tool_ena) contains configuration files for tool-on tests, i.e. only one tools is enabled.
- [tool_dis](tool_dis) contain configuration files for tool-off tests, i.e. all tools enabled except one selected.

Configuration files for a module of bitrate matcher which responsible also for setting `beta` (quality parameter) in a case of fixed betas locates in a directory [BRM](BRM/README.md). 

Lists with `beta`s are in [betas](betas/README.md) directory.

The user can set tool's configuration of tools individually for each image and each rate point:
- a directory [per-image](per-image) has configuration for each image. A configuration file should have the same name as the target image file. All parameters from the configuration file will overwrite parameters of the current configuration.
- a directory [per-image-per-bpp](per-image-per-bpp) has sub-directoris with the same name as the target image and files in it should have name `bpp<BPP>.json` where `<BPP>` is a target bpp of the image (multiply by 100), i.e. `bpp50.json` for 0.5 bpp.

## Configuration files for evaluation tool

Directory [`profiles`](profiles) has files with description of analyses/synthesis network of the codec for different operation point.