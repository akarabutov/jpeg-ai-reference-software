setup_system:
	sudo ./scripts/setup_system.sh

setup_env:
	./scripts/setup_env.sh

configure: setup_system setup_env

build_test_libs:
	./scripts/build_test_libs.sh

build_libs: build_test_libs

download_dvc_cache:
	./scripts/sFTP_mirror/download_cache.sh

download_test_ds:
	./scripts/download_test_ds.sh

download_models:
	./scripts/download_models.sh

download_train_ds:
	./scripts/download_train_ds.sh

test:
	python -m src.reco.scripts.eval --coding_type enc_dec --out_dir results/test

train:
	./scripts/train.sh

all: configure build_test_libs download_test_ds test

base_cfgs_img30:
	rm -Rf ./results/base_cfgs_img30
	python scripts/run_eval_script.py ./results/base_cfgs_img30 --cfg ./cfg/eval/tools_onoff_enc.json --imgs 00030_TE_560x888_8bit_sRGB.png
	python scripts/run_eval_script.py ./results/base_cfgs_img30 --cfg  ./cfg/eval/tools_onoff_dec.json --imgs 00030_TE_560x888_8bit_sRGB.png
	python scripts/merge_op_results.py ./results/base_cfgs_img30 --start-row 148 --template ./docs/template_img30.xlsm

base_cfgs:
	rm -Rf ./results/base_cfgs
	python scripts/run_eval_script.py ./results/base_cfgs --cfg ./cfg/eval/tools_onoff_enc.json
	python scripts/run_eval_script.py ./results/base_cfgs --cfg ./cfg/eval/tools_onoff_dec.json
	python scripts/merge_op_results.py ./results/base_cfgs

tool_ena:
	rm -Rf ./results/tool_ena
	python ./scripts/run_tool_perf.py ./cfg/tool_ena ./results/tool_ena

tool_dis:
	rm -Rf ./results/tool_dis
	python ./scripts/run_tool_perf.py ./cfg/tool_dis ./results/tool_dis


tool_perf: tool_ena tool_dis

unittest:
	CUDA_VISIBLE_DEVICES="-1" python -m unittest -v

build_docker:
	docker build . -t diveraak/jpeg_ai:latest

run_docker:
	docker run -it --mount src=.,target=/root/vm_init,type=bind diveraak/jpeg_ai:latest /bin/bash

export_models:
	# Export models without post-filters
	./scripts/export_models.sh