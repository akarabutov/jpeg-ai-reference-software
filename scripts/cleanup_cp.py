import argparse

import torch


def update_cp(cp, str_exclude):
    ans = dict()
    for k, v in cp.items():
        for exc_v in str_exclude:
            if k == exc_v:
                v = None
                k = None
        if k is not None:
            if isinstance(v, dict):
                v = update_cp(v, str_exclude)
            ans[k] = v
    return ans

def cleanup_cp(input_fn, output_fn, exclude):
    inp = torch.load(input_fn)

    ans = update_cp(inp, exclude)

    torch.save(ans, output_fn)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fn', type=str, default=None)
    parser.add_argument('output_fn', type=str, default=None)
    parser.add_argument('--exclude',
                        type=str,
                        nargs='+',
                        default=[],
                        help='List of keys for exclusion')
    args = parser.parse_args()
    cleanup_cp(args.input_fn, args.output_fn, args.exclude)
    print(f'Store data to file {args.output_fn}')


if __name__ == '__main__':
    main()
