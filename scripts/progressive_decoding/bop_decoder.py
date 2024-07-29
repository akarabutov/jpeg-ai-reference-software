import os
import sys
import torch
file_abs_path = os.path.abspath(__file__)
sys.path.append(file_abs_path)
from channel_list import sorted_channel_ids_dict

def reorder_weight_and_bias(ckpt_name, model_key, out_ckpt_name):
    layer_name = 'first_stage.0.0.conv1'
    state_dict = torch.load(ckpt_name)
    if layer_name + ".weight" in state_dict:
        state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][:, sorted_channel_ids_dict[model_key], :, :]
        state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][sorted_channel_ids_dict[model_key], :, :, :]
    if layer_name + ".bias" in state_dict:
        state_dict[layer_name + ".bias"] = state_dict[layer_name + ".bias"][sorted_channel_ids_dict[model_key]]

    layer_name = 'first_stage.1'
    if layer_name + ".weight" in state_dict:
        state_dict[layer_name + ".weight"] = state_dict[layer_name + ".weight"][sorted_channel_ids_dict[model_key], :, :, :]
    torch.save(state_dict, out_ckpt_name)

ckpt_names = ["decoder_Y_0.002.pth", "decoder_Y_0.012.pth", "decoder_Y_0.075.pth", "decoder_Y_0.5.pth"]
model_keys = ["Y_0.002", "Y_0.012", "Y_0.075", "Y_0.5"]
out_ckpt_names = ["decoder_Y_0.002.pth", "decoder_Y_0.012.pth", "decoder_Y_0.075.pth", "decoder_Y_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    model_key = model_keys[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_weight_and_bias(ckpt_name, model_key, out_ckpt_name)

def reorder_weight_and_bias_y2uv(ckpt_name, model_key, out_ckpt_name):
    state_dict = torch.load(ckpt_name)
    layer_name = 'first_stage.conv1' # decoder_UV
    if layer_name + ".weight" in state_dict:
        param = state_dict[layer_name + ".weight"].clone()
        #for item in range(param.shape[1] - 160, param.shape[1]):
        param[:, 0:160, :, :] = state_dict[layer_name + ".weight"][:, sorted_channel_ids_dict[model_key], :, :]
        state_dict[layer_name + ".weight"] = param

    torch.save(state_dict, out_ckpt_name)

ckpt_names = ["decoder_UV_0.002.pth", "decoder_UV_0.012.pth", "decoder_UV_0.075.pth", "decoder_UV_0.5.pth"]
model_keys = ["Y_0.002", "Y_0.012", "Y_0.075", "Y_0.5"]
out_ckpt_names = ["decoder_UV_0.002.pth", "decoder_UV_0.012.pth", "decoder_UV_0.075.pth", "decoder_UV_0.5.pth"]

for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    model_key = model_keys[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_weight_and_bias_y2uv(ckpt_name, model_key, out_ckpt_name)

def reorder_weight_and_bias_uv(ckpt_name, model_key, out_ckpt_name):
    state_dict = torch.load(ckpt_name)

    num_laten_chs = 96
    layer_name = "first_stage.conv1" # decoder_UV
    if layer_name + ".weight" in state_dict:
        param = state_dict[layer_name + ".weight"].clone()
        for idx in range(param.shape[1] - num_laten_chs, param.shape[1]):
            old_id = sorted_channel_ids_dict[model_key][idx - param.shape[1] + num_laten_chs]
            old_id += param.shape[1] - num_laten_chs
            param[:, idx, :, :] = state_dict[layer_name + ".weight"][:, old_id, :, :] 
        state_dict[layer_name + ".weight"] = param

    layer_name = "conv2_t"
    if layer_name + ".weight" in state_dict:
        param = state_dict[layer_name + ".weight"].clone()
        for idx in range(param.shape[0] - num_laten_chs, param.shape[0]):
            old_id = sorted_channel_ids_dict[model_key][idx - param.shape[0] + num_laten_chs]
            old_id += param.shape[0] - num_laten_chs
            param[idx, :, :, :] = state_dict[layer_name + ".weight"][old_id, :, :, :] 
        state_dict[layer_name + ".weight"] = param

    torch.save(state_dict, out_ckpt_name)

ckpt_names = ["decoder_UV_0.002.pth", "decoder_UV_0.012.pth", "decoder_UV_0.075.pth", "decoder_UV_0.5.pth"]
model_keys = ["UV_0.002", "UV_0.012", "UV_0.075", "UV_0.5"]
out_ckpt_names = ["decoder_UV_0.002.pth", "decoder_UV_0.012.pth", "decoder_UV_0.075.pth", "decoder_UV_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    model_key = model_keys[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_weight_and_bias_uv(ckpt_name, model_key, out_ckpt_name)
