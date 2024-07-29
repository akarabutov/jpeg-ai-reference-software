import argparse
import os

import torch

from src.codec.common.utils import update_dict_recursively


def merge_checkpoints(remove_list, exclude_list, *args):
    ans = dict()
    for cp_fn in args:
        print(f'Process file {cp_fn}...')
        cp = torch.load(cp_fn)
        print(f'{cp_fn} {cp.keys()}')
        # Remove keys
        for rk in remove_list:
            if rk in cp.keys():
                print(f'Remove key {rk} from cp in file {cp_fn}')
                del cp[rk]

        # Exclude blocks
        if len(ans) != 0:
            for e in exclude_list:
                if e in cp.keys():
                    print(f'Remove key {e} from cp in file {cp_fn}')
                    del cp[e]
        ans = update_dict_recursively(ans, cp)
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument('checkpoints', type=str, nargs='+', default=[])
    parser.add_argument('--exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of modeles on first level, which will be excluded from merging')
    parser.add_argument('--remove',
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of keys for removing')
    args = parser.parse_args()

    merged_cp = merge_checkpoints(args.remove, args.exclude, *args.checkpoints)

    output_dir = os.path.dirname(args.output_fn)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(merged_cp, args.output_fn)
    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
