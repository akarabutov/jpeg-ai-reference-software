import argparse
import os

import torch
from typing import List, Dict, Any


def update_dict_recursively(orig_dict: Dict[str, Any], new_dict: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in new_dict.items():
        if isinstance(val, dict):
            tmp = update_dict_recursively(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def update_element(src: Dict, dst: Dict, elem_url: List):
    elem_key = elem_url[0]
    concat_el = ".".join(elem_url)
    if len(elem_url) == 1 or (concat_el in src):
        if elem_key in src:
            dst[elem_key] = update_dict_recursively(src[elem_key], dst[elem_key]) 
        else:
            for k in src.keys():
                if k.startswith(elem_key):
                    if isinstance(src[k], dict):
                        dst[k] = update_dict_recursively(src[k], dst[k]) 
                    else:
                        dst[k] = src[k]
    else:
        update_element(src[elem_key], dst[elem_key], elem_url[1:])

def copy_checkpoints(input1_fn: str, input2_fn: str, output_fn: str, copy_list: List) -> None:
    input1_cp = torch.load(input1_fn)
    ans = torch.load(input2_fn)
    
    for el in copy_list:
        print(f"Copy {el} element(s).")
        update_element(input1_cp, ans, el.split("."))
    
    torch.save(ans, output_fn)


def main():
    parser = argparse.ArgumentParser(description=r'Copy elements from one checkpoints to another one')
    parser.add_argument('input1_fn', type=str, default=None)
    parser.add_argument('input2_fn', type=str, default=None)
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument('--copy',
                        type=str,
                        nargs='+',
                        default=[],
                        help=r'List of elements to be copied from a file "input1_fn" to structure of "input2_fn" and all of them will be stored to "output_fn". If the "input2_fn" have elements with the same names, they will be overwritten.')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_fn)
    os.makedirs(output_dir, exist_ok=True)

    copy_checkpoints(args.input1_fn, args.input2_fn, args.output_fn, args.copy)

    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
