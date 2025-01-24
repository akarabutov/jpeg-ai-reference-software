
import os
import torch
from src.codec.common import Decisions, ImageIO
from argparse import ArgumentParser

class CommonDump:
    @staticmethod
    def store_tensor_recurrently(base_path: str, latent_dict: dict, filename: str, decisions: Decisions, output_fmt: str, pgx_float_scale_factor: float) -> None:
        for n, v in latent_dict.items():
            path = os.path.join(base_path, n)
            if len(v) == 0:
                if isinstance(decisions[n], torch.Tensor):
                    os.makedirs(path, exist_ok=True)
                    if output_fmt == "pytorch":
                        torch.save(decisions[n], os.path.join(path, filename))
                    elif output_fmt == "pgx":
                        ImageIO.write_pgx(os.path.join(path, f"{filename}.pgx"), decisions[n], pgx_float_scale_factor)
                    else:
                        raise ValueError("Unsupported output data format")
                else:
                    raise ValueError("Type of decision isn't torch tensor")
            else:
                if n in decisions:
                    CommonDump.store_tensor_recurrently(path, v, filename, decisions[n], output_fmt, pgx_float_scale_factor)
                else:
                    raise ValueError(f"Decision doesn't have {n} item")
    
    @staticmethod
    def get_latents_dict(latents_list):
        ans = dict()
        for l in latents_list:
            l_sublist = l.split(".")
            cur_ans = ans
            for item in l_sublist:
                if item not in cur_ans:
                    cur_ans[item] = dict()
                cur_ans = cur_ans[item]
        return ans
                
def def_dump_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--latents_list',
                      type=str,
                      nargs="+",
                      default=['model_y.z_hat', 'model_y.y_hat', 'model_uv.z_hat', 'model_uv.y_hat'],
                      help='List of tensors for dumping')
    parser.add_argument('--output_format', type=str, default='pytorch', choices=['pytorch', 'pgx'], help=r'Output format of the data')
    parser.add_argument('--pgx_float_scale_factor', type=float, default=1.0, help=r'Sclae factor to store floating point data in PGX files')
    
    return parser