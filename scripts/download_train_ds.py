# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
"""Download and unpack the JPEG AI training and validation datasets.

The datasets live on the JPEG AI sFTP server; the account password is distributed through the
ISO document referenced by ``--help``.  This script fetches the archives, unpacks them into the
layout ``scripts/train.sh`` expects and writes the file lists the training code reads:

    data/jpegai_training_random_crop/          training patches
    data/jpegai_training_random_crop/jpegai_training_set512_random_crop_16.txt
    data/jpegai_validation_set/                validation images
    data/jpegai_validation_set/jpegai_validation_set_10.txt

Typical use::

    python scripts/download_train_ds.py                 # everything, asks for the password
    python scripts/download_train_ds.py --parts train   # training patches only
    python scripts/download_train_ds.py --status        # what is already on disk
    python scripts/download_train_ds.py --dry-run       # show what would be run

Transfers resume: an archive that is already complete is skipped and a partial one continues
where it stopped, so an interrupted download can simply be started again.  The password is
never passed on a command line -- it is read from ``--password-file``, from the
``JPEG_AI_SFTP_PASSWORD`` environment variable or interactively, and handed to ``sshpass``
through the environment.
"""

import argparse
import getpass
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_HOST = 'amalia.img.lx.it.pt'
DEFAULT_USER = 'jpeg-ai'
PASSWORD_ENV = 'JPEG_AI_SFTP_PASSWORD'
PASSWORD_DOC = (
    'https://sd.iso.org/documents/ui/#!/browse/iso/iso-iec-jtc-1/iso-iec-jtc-1-sc-29/'
    'iso-iec-jtc-1-sc-29-wg-1/library/6/98-Sydney/OUTPUT%20N-documents/'
    'wg1n100422-098-ICQ-Access%20information%20for%20JPEG%20AI%20dataset')

TRAIN_DIR = 'jpegai_training_random_crop'
VALID_DIR = 'jpegai_validation_set'
TRAIN_LIST = 'jpegai_training_set512_random_crop_16.txt'
VALID_LIST = 'jpegai_validation_set_10.txt'

NATURAL_REMOTE_DIR = '/train_and_valid_natural/cropped'
SCC_REMOTE_DIR = '/train_and_valid_scc700'

# Rough size of the extracted content relative to the downloaded archives; used only to warn
# about a disk that is too small before a multi-hour transfer starts.
EXTRACTED_SIZE_FACTOR = 1.2


class DatasetError(Exception):
    """A user-facing error: reported as a message, never as a traceback."""


class Archive:
    """One remote archive and what to do with it once it has been downloaded."""

    def __init__(self, name, remote_dir, part, extract_to, members=(), flatten=True):
        self.name = name
        self.remote_dir = remote_dir
        self.part = part
        self.extract_to = extract_to
        self.members = tuple(members)
        self.flatten = flatten

    @property
    def is_tar(self):
        return self.name.endswith(('.tar', '.tar.gz', '.tgz'))

    def __repr__(self):
        return f'Archive({self.name})'


# The archive names published for the DIS dataset.  `--refresh` replaces this list with what
# the server actually offers, so a renamed or added archive does not need a code change.
ARCHIVES = [
    Archive('jpegai_training_random_crop_00000-01299.zip', NATURAL_REMOTE_DIR, 'train',
            TRAIN_DIR, members=('*.png', )),
    Archive('jpegai_training_random_crop_01300-02599.zip', NATURAL_REMOTE_DIR, 'train',
            TRAIN_DIR, members=('*.png', )),
    Archive('jpegai_training_random_crop_02600-03899.zip', NATURAL_REMOTE_DIR, 'train',
            TRAIN_DIR, members=('*.png', )),
    Archive('jpegai_training_random_crop_03900-5263.zip', NATURAL_REMOTE_DIR, 'train',
            TRAIN_DIR, members=('*.png', )),
    Archive('jpegai_validation_set.zip', NATURAL_REMOTE_DIR, 'validation', VALID_DIR,
            members=('*.png', '*.txt')),
    Archive('scc7000_patchs2.tar', SCC_REMOTE_DIR, 'scc', TRAIN_DIR, flatten=False),
]

PARTS = ('train', 'scc', 'validation')


# ######################################################################################################################
#  Remote archive discovery
# ######################################################################################################################
# `ls -l` over sftp prints a long listing: mode, links, owner, group, size, date, name.
LS_LINE_RE = re.compile(r'^[-bcdlps][-rwxSsTt]{9}[.+]?\s+\d+\s+\S+\s+\S+\s+(?P<size>\d+)\s+'
                        r'\S+\s+\S+\s+\S+\s+(?P<name>.+)$')


def parse_ls_output(text):
    """Turn the ``ls -l`` output of an sftp session into ``{basename: size}``."""
    ans = dict()
    for line in text.splitlines():
        match = LS_LINE_RE.match(line.strip())
        if match is None:
            continue
        name = os.path.basename(match.group('name').strip())
        if name in ('.', '..') or not name:
            continue
        ans[name] = int(match.group('size'))
    return ans


def classify_archive(name, remote_dir):
    """Decide which dataset an archive belongs to and where its content should go."""
    lowered = name.lower()
    if remote_dir == SCC_REMOTE_DIR or 'scc' in lowered:
        return Archive(name, remote_dir, 'scc', TRAIN_DIR, flatten=False)
    if 'valid' in lowered:
        return Archive(name, remote_dir, 'validation', VALID_DIR, members=('*.png', '*.txt'))
    if 'train' in lowered:
        return Archive(name, remote_dir, 'train', TRAIN_DIR, members=('*.png', ))
    return None


def refresh_archives(listings, parts):
    """Rebuild the archive list from what the server actually offers.

    Keeps the script working when an archive is renamed, split differently or added, at the
    price of guessing each archive's role from its name.
    """
    ans = list()
    for remote_dir, listing in sorted(listings.items()):
        for name in sorted(listing):
            if not name.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
                continue
            archive = classify_archive(name, remote_dir)
            if archive is not None and archive.part in parts:
                ans.append(archive)
    return ans


def select_archives(archives, parts):
    return [a for a in archives if a.part in parts]


# ######################################################################################################################
#  Running sftp
# ######################################################################################################################
class SftpClient:
    """Drives ``sftp``, optionally through ``sshpass``, feeding commands on standard input.

    Commands go to the child's stdin rather than to ``sftp -b`` on purpose: batch mode turns
    on ``BatchMode=yes``, which disables password authentication and would defeat ``sshpass``.
    """

    def __init__(self, host, user, password=None, identity=None, accept_host_key=False,
                 ssh_options=(), dry_run=False, quiet=False):
        self.host = host
        self.user = user
        self.password = password
        self.identity = identity
        self.accept_host_key = accept_host_key
        self.ssh_options = list(ssh_options)
        self.dry_run = dry_run
        self.quiet = quiet

    def argv(self):
        argv = list()
        if self.password is not None:
            # `-e` reads the password from SSHPASS, so it never appears in the process list.
            argv += ['sshpass', '-e']
        argv.append('sftp')
        if self.identity is not None:
            argv += ['-i', self.identity]
        if self.accept_host_key:
            argv += ['-o', 'StrictHostKeyChecking=accept-new']
        for option in self.ssh_options:
            argv += ['-o', option]
        argv.append(f'{self.user}@{self.host}')
        return argv

    def env(self):
        env = dict(os.environ)
        if self.password is not None:
            env['SSHPASS'] = self.password
        else:
            env.pop('SSHPASS', None)
        return env

    def run(self, commands, cwd=None, capture=False):
        """Run one sftp session; ``commands`` is the list of sftp commands to feed it."""
        argv = self.argv()
        script = '\n'.join(list(commands) + ['bye', ''])
        if self.dry_run:
            print('$ {} <<EOF'.format(' '.join(shlex.quote(x) for x in argv)))
            for line in commands:
                print(f'    {line}')
            print('EOF')
            return '', 0

        proc = subprocess.Popen(argv,
                                cwd=cwd,
                                env=self.env(),
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True)
        chunks = list()
        try:
            try:
                proc.stdin.write(script)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                # sftp gave up before reading the commands, typically on a failed login;
                # its message is still on stdout and is reported by the caller.
                pass
            for line in proc.stdout:
                chunks.append(line)
                if not capture and not self.quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        finally:
            proc.wait()
        return ''.join(chunks), proc.returncode

    def listing(self, remote_dir):
        """Return ``{name: size}`` for the archives in ``remote_dir``."""
        output, code = self.run([f'cd {remote_dir}', 'ls -l'], capture=True)
        if self.dry_run:
            return dict()
        if code != 0:
            detail = ' '.join(output.split())[-300:] or f'sftp exited with code {code}'
            raise DatasetError(
                f'Could not list {remote_dir} on {self.host}: {detail}\n'
                'Check the password (see --help for where it is published), the host key '
                '(--accept-host-key on first connection) and network access to the server.')
        return parse_ls_output(output)


# ######################################################################################################################
#  Download
# ######################################################################################################################
def local_archive_path(archives_dir, archive):
    return os.path.join(archives_dir, archive.name)


def needs_download(path, expected_size):
    """True when the archive is missing or shorter than the copy on the server."""
    if not os.path.isfile(path):
        return True
    if expected_size is None:
        return False
    return os.path.getsize(path) < expected_size


def download_archives(client, archives, archives_dir, sizes, retries=3, dry_run=False):
    """Fetch every archive that is not already complete, resuming partial transfers.

    ``reget`` continues a partial file, so an interrupted run costs only the bytes that were
    still missing.  Returns the list of archives that could not be completed.
    """
    failed = list()
    by_dir = dict()
    for archive in archives:
        by_dir.setdefault(archive.remote_dir, list()).append(archive)

    for remote_dir, group in sorted(by_dir.items()):
        pending = list()
        for archive in group:
            path = local_archive_path(archives_dir, archive)
            expected = sizes.get(archive.name)
            if dry_run or needs_download(path, expected):
                pending.append(archive)
            else:
                print(f'  {archive.name}: already complete ({format_size(os.path.getsize(path))})')
        if not pending:
            continue

        for archive in pending:
            expected = sizes.get(archive.name)
            path = local_archive_path(archives_dir, archive)
            done = False
            for attempt in range(1, retries + 1):
                suffix = '' if attempt == 1 else f' (attempt {attempt})'
                print(f'  {archive.name}: downloading{suffix}')
                client.run([f'cd {remote_dir}', f'reget {archive.name}'], cwd=archives_dir)
                if dry_run or not needs_download(path, expected):
                    done = True
                    break
                if attempt < retries:
                    delay = 2 ** attempt
                    print(f'  {archive.name}: incomplete, resuming in {delay}s')
                    time.sleep(delay)
            if not done:
                failed.append(archive)
    return failed


# ######################################################################################################################
#  Extraction
# ######################################################################################################################
def extract_argv(archive, archive_path, dest_dir):
    """Build the command that unpacks one archive.

    Zip archives are flattened (``-j``) into the dataset directory and filtered to the members
    that matter, which keeps the result identical no matter how a given archive nests its
    content.  The SCC tar keeps its internal directory, matching the paths in
    ``cfg/training_list/Q*_training_list.txt``.
    """
    if archive.is_tar:
        return ['tar', '-xf', archive_path, '-C', dest_dir]
    argv = ['unzip', '-o', '-q']
    if archive.flatten:
        argv.append('-j')
    argv.append(archive_path)
    argv += list(archive.members)
    argv += ['-d', dest_dir]
    return argv


def run_command(argv, dry_run=False, quiet=False):
    printable = ' '.join(shlex.quote(x) for x in argv)
    if dry_run:
        print(f'$ {printable}')
        return 0
    if not quiet:
        print(f'  $ {printable}')
    return subprocess.call(argv)


def extract_archives(archives, archives_dir, data_dir, dry_run=False, quiet=False):
    """Unpack the archives; returns the list of archives whose extraction failed."""
    failed = list()
    for archive in archives:
        archive_path = local_archive_path(archives_dir, archive)
        if not dry_run and not os.path.isfile(archive_path):
            print(f'  {archive.name}: not downloaded, skipping extraction')
            failed.append(archive)
            continue
        dest_dir = os.path.join(data_dir, archive.extract_to)
        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
        code = run_command(extract_argv(archive, archive_path, dest_dir), dry_run=dry_run,
                           quiet=quiet)
        if code != 0:
            print(f'  {archive.name}: extraction failed with exit code {code}')
            failed.append(archive)
    return failed


# ######################################################################################################################
#  File lists
# ######################################################################################################################
def collect_images(root):
    """Every PNG under ``root``, as sorted paths relative to it.

    Walking recursively covers both layouts in play: the flattened crops from the zip archives
    and the SCC patches, which keep their own directory inside the tar.
    """
    ans = list()
    for dir_path, _, files in os.walk(root):
        rel_dir = os.path.relpath(dir_path, root)
        for name in files:
            if not name.lower().endswith('.png'):
                continue
            rel = name if rel_dir == '.' else os.path.join(rel_dir, name)
            ans.append(rel.replace(os.sep, '/'))
    return sorted(ans)


def write_list_file(path, names, dry_run=False):
    if dry_run:
        print(f'$ write {len(names)} entries to {path}')
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for name in names:
            f.write(f'{name}\n')


def generate_lists(data_dir, parts, force=False, dry_run=False):
    """Write the file lists the training scripts read, when they are missing."""
    made = list()

    if 'train' in parts or 'scc' in parts:
        train_root = os.path.join(data_dir, TRAIN_DIR)
        list_path = os.path.join(train_root, TRAIN_LIST)
        if dry_run:
            print(f'$ regenerate {list_path}')
        elif os.path.isdir(train_root):
            if force or not os.path.isfile(list_path):
                names = collect_images(train_root)
                write_list_file(list_path, names)
                made.append((list_path, len(names)))

    if 'validation' in parts:
        valid_root = os.path.join(data_dir, VALID_DIR)
        list_path = os.path.join(valid_root, VALID_LIST)
        if dry_run:
            print(f'$ write {list_path} if the archive did not ship it')
        elif os.path.isdir(valid_root):
            # The validation archive ships its own list; only step in when it is absent.
            if force or not os.path.isfile(list_path):
                names = collect_images(valid_root)
                write_list_file(list_path, names)
                made.append((list_path, len(names)))

    return made


# ######################################################################################################################
#  Status
# ######################################################################################################################
def format_size(num_bytes):
    if num_bytes is None:
        return '-'
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024.0 or unit == 'TiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024.0
    return f'{value:.1f} TiB'


def display_path(path):
    """Show a path relative to the working directory when that is the shorter form."""
    relative = os.path.relpath(path, os.getcwd())
    return relative if not relative.startswith(os.pardir) else path


def format_table(headers, rows):
    cols = [[str(h)] + [str(r[i]) for r in rows] for i, h in enumerate(headers)]
    widths = [max(len(x) for x in col) for col in cols]
    out = ['  '.join(str(h).ljust(w) for h, w in zip(headers, widths)),
           '  '.join('-' * w for w in widths)]
    for row in rows:
        out.append('  '.join(str(c).ljust(w) for c, w in zip(row, widths)))
    return '\n'.join(out)


def print_status(data_dir, archives_dir, archives, sizes=None):
    sizes = sizes or dict()
    rows = list()
    for archive in archives:
        path = local_archive_path(archives_dir, archive)
        expected = sizes.get(archive.name)
        if not os.path.isfile(path):
            state = 'missing'
            local = '-'
        else:
            local = format_size(os.path.getsize(path))
            if expected is None:
                state = 'present'
            elif os.path.getsize(path) < expected:
                state = 'partial'
            else:
                state = 'complete'
        rows.append([archive.part, archive.name, state, local, format_size(expected)])
    print('Archives in {}:'.format(display_path(archives_dir)))
    print(format_table(['part', 'archive', 'state', 'local', 'remote'], rows))

    rows = list()
    for name, list_name in ((TRAIN_DIR, TRAIN_LIST), (VALID_DIR, VALID_LIST)):
        root = os.path.join(data_dir, name)
        if os.path.isdir(root):
            images = collect_images(root)
            list_path = os.path.join(root, list_name)
            listed = '-'
            if os.path.isfile(list_path):
                with open(list_path, 'r') as f:
                    listed = str(sum(1 for line in f if line.strip()))
            rows.append([name, str(len(images)), listed])
        else:
            rows.append([name, 'absent', '-'])
    print('\nUnpacked datasets in {}:'.format(display_path(data_dir)))
    print(format_table(['directory', 'images', 'entries in list'], rows))


def check_disk_space(path, archives, sizes, skip=False):
    """Warn before a long transfer when the target filesystem is too small."""
    known = [sizes[a.name] for a in archives if a.name in sizes]
    if skip or not known:
        return
    needed = int(sum(known) * (1.0 + EXTRACTED_SIZE_FACTOR))
    free = shutil.disk_usage(path).free
    print(f'Archives to fetch: {format_size(sum(known))}; '
          f'about {format_size(needed)} needed with the unpacked copy, '
          f'{format_size(free)} free')
    if free < needed:
        raise DatasetError(
            f'Not enough free space in {path}: {format_size(free)} available, about '
            f'{format_size(needed)} needed. Free some space, point --data-dir at a larger '
            'filesystem, fetch fewer --parts, or pass --no-space-check to continue anyway.')


# ######################################################################################################################
#  Tools and credentials
# ######################################################################################################################
TOOL_PACKAGES = {
    'sftp': 'openssh-client',
    'sshpass': 'sshpass',
    'unzip': 'unzip',
    'tar': 'tar',
}


def require_tools(names, warn_only=False):
    """Check that the external tools this run needs are installed."""
    missing = [x for x in names if shutil.which(x) is None]
    if not missing:
        return
    packages = sorted({TOOL_PACKAGES.get(x, x) for x in missing})
    message = ('Missing required tool(s): {}. Install them with '
               '"sudo apt install {}".'.format(', '.join(missing), ' '.join(packages)))
    if warn_only:
        print(f'warning: {message}')
        return
    raise DatasetError(message)


def read_password(args):
    """Resolve the sFTP password without ever putting it on a command line."""
    if args.identity is not None:
        return None
    if args.password_file is not None:
        with open(args.password_file, 'r') as f:
            password = f.read().strip()
        if not password:
            raise DatasetError(f'{args.password_file} is empty')
        return password
    password = os.environ.get(PASSWORD_ENV)
    if password:
        return password
    if args.dry_run:
        return ''
    if not sys.stdin.isatty():
        raise DatasetError(
            f'No password available: set {PASSWORD_ENV}, pass --password-file, or run the '
            'script on a terminal so it can prompt.')
    print(f'The sFTP password is published here:\n  {PASSWORD_DOC}')
    return getpass.getpass('Password: ')


# ######################################################################################################################
#  main
# ######################################################################################################################
def build_parser():
    parser = argparse.ArgumentParser(
        prog='download_train_ds',
        description='Download and unpack the JPEG AI training and validation datasets.',
        epilog=f'The sFTP password is published in the ISO document: {PASSWORD_DOC}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--parts', nargs='+', choices=list(PARTS) + ['all'], default=['all'],
                        help='Which datasets to fetch: the natural training crops, the SCC '
                             'patches, and/or the validation set')
    parser.add_argument('--data-dir', dest='data_dir', default=os.path.join(REPO_ROOT, 'data'),
                        help='Directory the datasets are unpacked into')
    parser.add_argument('--archives-dir', dest='archives_dir', default=None,
                        help='Where the downloaded archives are kept (default: <data-dir>)')

    parser.add_argument('--host', default=DEFAULT_HOST, help='sFTP host')
    parser.add_argument('--user', default=DEFAULT_USER, help='sFTP user')
    parser.add_argument('--password-file', dest='password_file', default=None,
                        help='File holding the sFTP password')
    parser.add_argument('--identity', default=None,
                        help='Authenticate with this ssh private key instead of a password')
    parser.add_argument('--accept-host-key', dest='accept_host_key', action='store_true',
                        help='Accept an unknown host key on first connection')
    parser.add_argument('--ssh-option', dest='ssh_options', action='append', default=list(),
                        metavar='OPTION', help='Extra "-o" option for ssh; repeatable')
    parser.add_argument('--retries', type=int, default=3,
                        help='Attempts per archive before giving up')

    parser.add_argument('--refresh', action='store_true',
                        help='Take the archive list from the server instead of the built-in one')
    parser.add_argument('--status', action='store_true',
                        help='Report what is already on disk and exit, without connecting')
    parser.add_argument('--list-remote', dest='list_remote', action='store_true',
                        help='Print the archives the server offers and exit')
    parser.add_argument('--skip-download', dest='skip_download', action='store_true',
                        help='Unpack archives that are already downloaded')
    parser.add_argument('--skip-extract', dest='skip_extract', action='store_true',
                        help='Download only, do not unpack')
    parser.add_argument('--remove-archives', dest='remove_archives', action='store_true',
                        help='Delete each archive once it has been unpacked')
    parser.add_argument('--force-lists', dest='force_lists', action='store_true',
                        help='Rewrite the training and validation file lists even if present')
    parser.add_argument('--no-space-check', dest='no_space_check', action='store_true',
                        help='Do not refuse to start when free disk space looks insufficient')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        help='Print what would be done and exit')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not echo the sftp session')
    return parser


def resolve_parts(parts):
    return set(PARTS) if 'all' in parts else set(parts)


def main(argv=None):
    args = build_parser().parse_args(argv)
    parts = resolve_parts(args.parts)
    data_dir = os.path.abspath(args.data_dir)
    archives_dir = os.path.abspath(args.archives_dir or data_dir)

    archives = select_archives(ARCHIVES, parts)

    if args.status:
        if not os.path.isdir(data_dir):
            raise DatasetError(f'Data directory does not exist: {data_dir}')
        print_status(data_dir, archives_dir, archives)
        return 0

    if not args.dry_run:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(archives_dir, exist_ok=True)

    needs_network = args.list_remote or not args.skip_download
    password = read_password(args) if needs_network else None
    needed = list()
    if needs_network:
        needed += ['sftp'] + (['sshpass'] if password else [])
    if not args.skip_extract:
        needed += sorted({'tar' if a.is_tar else 'unzip' for a in archives})
    require_tools(needed, warn_only=args.dry_run)

    client = SftpClient(host=args.host,
                        user=args.user,
                        password=password or None,
                        identity=args.identity,
                        accept_host_key=args.accept_host_key,
                        ssh_options=args.ssh_options,
                        dry_run=args.dry_run,
                        quiet=args.quiet)

    sizes = dict()
    if args.list_remote or args.refresh or not (args.skip_download or args.dry_run):
        listings = dict()
        for remote_dir in sorted({a.remote_dir for a in archives}):
            listings[remote_dir] = client.listing(remote_dir)
            sizes.update(listings[remote_dir])

        if args.list_remote:
            for remote_dir, listing in sorted(listings.items()):
                rows = [[name, format_size(size)] for name, size in sorted(listing.items())]
                print(f'\n{remote_dir}:')
                print(format_table(['name', 'size'], rows) if rows else '  (empty)')
            return 0

        if args.refresh and sizes:
            refreshed = refresh_archives(listings, parts)
            if refreshed:
                archives = refreshed
                print('Archives offered by the server:')
                for archive in archives:
                    print(f'  {archive.part:<10} {archive.name} '
                          f'({format_size(sizes.get(archive.name))})')

    if not args.skip_download:
        check_disk_space(archives_dir, archives, sizes, skip=args.no_space_check or args.dry_run)
        print('\nDownloading archives:')
        failed = download_archives(client, archives, archives_dir, sizes,
                                   retries=max(1, args.retries), dry_run=args.dry_run)
        if failed:
            names = ', '.join(a.name for a in failed)
            raise DatasetError(f'Could not download: {names}. Run the script again to resume.')

    if not args.skip_extract:
        print('\nUnpacking archives:')
        failed = extract_archives(archives, archives_dir, data_dir, dry_run=args.dry_run,
                                  quiet=args.quiet)
        if failed:
            names = ', '.join(a.name for a in failed)
            raise DatasetError(f'Could not unpack: {names}')
        if args.remove_archives:
            removed = 0
            for archive in archives:
                path = local_archive_path(archives_dir, archive)
                if args.dry_run:
                    print(f'$ rm {path}')
                elif os.path.isfile(path):
                    os.remove(path)
                    removed += 1
            if removed:
                print(f'  removed {removed} unpacked archive(s); '
                      'they are reported as missing below')

        print('\nFile lists:')
        made = generate_lists(data_dir, parts, force=args.force_lists, dry_run=args.dry_run)
        for path, count in made:
            print(f'  {display_path(path)}: {count} entries')
        if not made and not args.dry_run:
            print('  already present')

    if not args.dry_run:
        print()
        print_status(data_dir, archives_dir, archives, sizes)
        print('\nTraining can now be started with scripts/train.sh')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except DatasetError as error:
        print(f'error: {error}', file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print('\ninterrupted; run the script again to resume', file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        # The output was piped into something that stopped reading, such as `head`.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
