# Configuration files for bitrate matcher

By default ([default.json](default.json)) bitrate matcher set quality parameter (`beta`) according to a target rate. 


To use bitrate matching functionality there are the following files:
- [regen_list.json](regen_list.json) is a configuration file for getting `beta` per each image and rate for matching the rate (up to 10%).
- [use_list.json](use_list.json) is a configuration file for using pregenerated list with `beta` (see [betas](../betas/README.md)).

