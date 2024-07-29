import os
import sys
import numpy
import torch
file_abs_path = os.path.abspath(__file__)
prog_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_abs_path)))
sys.path.append(os.path.join(prog_dir))

sys.path.append(file_abs_path)
from channel_list import sorted_channel_ids_dict

def reorder_channel_wise_entropy(ckpt_name, out_ckpt_name):
    state_dict = torch.load(ckpt_name)
    channel_wise_entropy = state_dict['channel_wise_entropy'].detach().cpu().numpy()
    sorted_ids = (numpy.argsort(channel_wise_entropy)[::-1]).tolist()
    state_dict['channel_wise_entropy'] = state_dict['channel_wise_entropy'][sorted_ids]

    torch.save(state_dict, out_ckpt_name)

ckpt_names = ["Y_0.002.pth", "Y_0.012.pth", "Y_0.075.pth", "Y_0.5.pth"]
model_keys = ["Y_0.002", "Y_0.012", "Y_0.075", "Y_0.5"]
out_ckpt_names = ["Y_0.002.pth", "Y_0.012.pth", "Y_0.075.pth", "Y_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_channel_wise_entropy(ckpt_name, out_ckpt_name)

ckpt_names = ["UV_0.002.pth", "UV_0.012.pth", "UV_0.075.pth", "UV_0.5.pth"]
model_keys = ["UV_0.002", "UV_0.012", "UV_0.075", "UV_0.5"]
out_ckpt_names = ["UV_0.002.pth", "UV_0.012.pth", "UV_0.075.pth", "UV_0.5.pth"]
for idx in range(len(ckpt_names)):
    ckpt_name = ckpt_names[idx]
    out_ckpt_name = out_ckpt_names[idx]
    reorder_channel_wise_entropy(ckpt_name, out_ckpt_name)
