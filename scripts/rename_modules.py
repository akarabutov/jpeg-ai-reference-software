import argparse

import torch
from typing import List

def update_cp_elem(cp, old_name: List, new_name: List):
    if len(old_name) == 0 or len(new_name) == 0:
        return
    old_n = old_name[0]
    new_n = new_name[0]
    if old_n != new_n:
        cp[new_n] = cp[old_n]
        del cp[old_n]
    concat_old_name = ".".join(old_name)
    if concat_old_name in cp:
        concat_new_name = ".".join(new_name)
        cp[concat_new_name] = cp[concat_old_name]
        del cp[concat_old_name]        
    else:
        update_cp_elem(cp[new_n], old_name[1:], new_name[1:])
    

def update_cp(cp, str_replaces):
    for os, ns in str_replaces.items():
        update_cp_elem(cp, os.split("."), ns.split("."))
    return cp


def list2dict(lst):
    assert len(lst) % 2 == 0
    ans = dict()
    for i in range(0, len(lst), 2):
        ans[lst[i]] = lst[i + 1]
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fn', type=str, default=None)
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument(
        '--rename',
        type=str,
        nargs='+',
        default=[],
        help='Patterns of names for renameing and the new values in form: <pattern> <new_name>')
    args = parser.parse_args()

    inp = torch.load(args.input_fn)

    a = list2dict(args.rename)
    print(a)
    ans = update_cp(inp, a)

    torch.save(ans, args.output_fn)
    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
