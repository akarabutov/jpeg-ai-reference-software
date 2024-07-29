import argparse
import os
import fnmatch
import sys
from cleanup_cp import cleanup_cp

def get_full_list_of_files(root_dir, mask, base_dir=None):
    ans = list()
    if base_dir is None:
        base_dir = root_dir
    for r, _, f in os.walk(root_dir):
        for fn in f:
            if fnmatch.fnmatch(fn, mask):
                ans.append(os.path.relpath(os.path.join(r, fn), base_dir))
    return ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default=None)
    parser.add_argument('--push', action='store_true', default=False)
    parser.add_argument('--pack', action='store_true', default=False)
    parser.add_argument('--ckeckpoints-mask', default='*.pth', help=r'Glob mask for checkpoints')
    parser.add_argument('-r', default=None, help='Name of remote of DVC for storing checkpints')
    parser.add_argument('--exclude-keys', default=["best_loss", "optimizer"], nargs="+", help='List of keys for exclusion from checkpoints before storing')
    args = parser.parse_args()

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    models_path = os.path.join(base_path, 'models', args.model_name)            
    
    cp_list = get_full_list_of_files(models_path, args.ckeckpoints_mask, base_dir=base_path)
    if len(args.exclude_keys) > 0 and len(args.exclude_keys[0]) > 0:
        for fn in cp_list:
            cleanup_cp(fn, fn, args.exclude_keys)
        
    dvc_list = [x + ".dvc" for x in cp_list]
       
    
    python_bin = os.path.dirname(sys.executable)
    dvc_path = os.path.join(python_bin, 'dvc')

    cmds = [
        f'{dvc_path} add ' + ' '.join(cp_list),
        f'{dvc_path} commit ' + ' '.join(dvc_list)
    ]
    if args.push:
        s = f'{dvc_path} push ' + ' '.join(dvc_list)
        if args.r is not None:
            s += f' -r {args.r}'
        cmds.append(s)

    os.system(' && '.join(cmds))
    os.system(f'git add ' + ' '.join(dvc_list))

    if args.pack:
        path = os.path.join(base_path, f'{args.model_name}.tgz')
        os.system(f'tar -cvzf {path} ' +  ' '.join(cp_list))


if __name__ == '__main__':
    main()
