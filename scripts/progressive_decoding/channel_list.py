import torch
import numpy
def get_sorted_channels(model_name):
    common_model_path = f"../VM_common_int/{model_name}.pth"
    state_dict = torch.load(common_model_path)
    channel_wise_entropy = state_dict['channel_wise_entropy'].detach().cpu().numpy()
    print(channel_wise_entropy)
    if numpy.sum(abs(channel_wise_entropy)) < 1e-5:
        ret = list(range(0, channel_wise_entropy.shape[0]))
    else:
        ret = (numpy.argsort(channel_wise_entropy)[::-1]).tolist()
    print(ret)
    return ret

sorted_channel_ids_dict = {

  "Y_0.075": get_sorted_channels("Y_0.075"),
  "Y_0.5": get_sorted_channels("Y_0.5"),
  "Y_0.002": get_sorted_channels("Y_0.002"),
  "Y_0.012": get_sorted_channels("Y_0.012"),

  "UV_0.075": get_sorted_channels("UV_0.075"),  
  "UV_0.5":  get_sorted_channels("UV_0.5"),
  "UV_0.002": get_sorted_channels("UV_0.002"),  
  "UV_0.012": get_sorted_channels("UV_0.012")

}
