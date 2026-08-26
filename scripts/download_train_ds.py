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
"""Download the JPEG AI training and validation datasets, asking what is needed.

Run without arguments to be guided through the choices:

    1. which mirror to download from -- ISO or ITU;
    2. which datasets -- training, validation or both;
    3. for training, whether the natural content comes as full-size images or as cropped
       patches, and whether the extra (screen content, high frequency, ...) datasets are
       wanted, each with the download size it adds;
    4. a final confirmation showing the total size.

Every answer also has a command-line flag, so the same run can be repeated unattended::

    python scripts/download_train_ds.py --source itu --datasets both \\
           --natural patches --extras all --yes

The catalogue of archives is read from the mirror itself: the script walks the published
directory index, works out what each archive holds from its name and reports real sizes taken
from the listing (or from a HEAD request). Nothing about the file names is hard-coded, so a
renamed or added archive is picked up automatically; `--list-remote` prints what was found and
how it was classified.

Downloads resume: a partial file continues with a Range request and a complete one is skipped.
Archives are unpacked into the layout the training scripts read:

    data/jpegai_training/                      full-size natural training images
    data/jpegai_training_random_crop/          training patches, extra datasets in their
                                               own subdirectories, as referenced by
                                               cfg/training_list/Q*_training_list.txt
    data/jpegai_validation_set/                validation images
"""

import argparse
import hashlib
import html.parser
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the datasets are published.  Both mirrors carry the same content.
MIRRORS = {
    'iso': {
        'title': 'ISO -- standards.iso.org',
        'url': 'https://standards.iso.org/iso-iec/6048/-3/ed-1/en/',
    },
    'itu': {
        'title': 'ITU -- www.itu.int',
        'url': 'https://www.itu.int/wftp3/Public/t/testsignal/SpeImage/T840-3/v2026_01/',
    },
}
DEFAULT_SOURCE = 'iso'

# Local layout expected by scripts/train.sh and cfg/train.json.
TRAIN_FULL_DIR = 'jpegai_training'
TRAIN_CROP_DIR = 'jpegai_training_random_crop'
VALID_DIR = 'jpegai_validation_set'
VALID_FULL_DIR = 'jpegai_validation_set_full'
TRAIN_LIST = 'jpegai_training_set512_random_crop_16.txt'
VALID_LIST = 'jpegai_validation_set_10.txt'

ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz', '.7z')
CHECKSUM_SUFFIXES = ('.md5', '.sha1', '.sha256')
# Large archives are published split into volumes: <name>.7z.001, .002 and so on.
VOLUME_RE = re.compile(r'^(?P<base>.+\.(?:7z|zip|tar|tar\.gz|tgz|tar\.bz2|tar\.xz))'
                       r'\.(?P<index>\d{2,})$', re.IGNORECASE)

# The catalogue published as v2026_01, the same on both mirrors.  It is only used when a
# mirror does not serve a directory index; the sizes are still confirmed against the
# server, and a file the server does not have is dropped.
KNOWN_CATALOGUE_VERSION = 'v2026_01'
KNOWN_CATALOGUE = (
    ('jpegai_reference_software.zip', 1494803328),
    ('jpegai_train-extra_CP50.tar', 3830272),
    ('jpegai_train-extra_EXCEL300.tar', 33928704),
    ('jpegai_train-extra_HF2000.tar', 1064664064),
    ('jpegai_train-extra_LQ7000.tar', 3267666944),
    ('jpegai_train-extra_MD300.tar', 73410560),
    ('jpegai_train-extra_PHF200.tar', 1207808),
    ('jpegai_train-extra_PHFA500.tar', 3316736),
    ('jpegai_train-extra_SCC7000P2.tar', 1409797632),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.001', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.002', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.003', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.004', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.005', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.006', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.007', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.008', 5368709120),
    ('jpegai_train-natural-cropped_00000-01299_bundle.7z.009', 4666395408),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.001', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.002', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.003', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.004', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.005', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.006', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.007', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.008', 5368709120),
    ('jpegai_train-natural-cropped_01300-02599_bundle.7z.009', 4202585266),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.001', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.002', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.003', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.004', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.005', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.006', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.007', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.008', 5368709120),
    ('jpegai_train-natural-cropped_02600-03899_bundle.7z.009', 3339274609),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.001', 5368709120),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.002', 5368709120),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.003', 5368709120),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.004', 5368709120),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.005', 5368709120),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.006', 5368709120),
    ('jpegai_train-natural-cropped_03900-05263_bundle.7z.007', 2444367678),
    ('jpegai_train-natural-full_bundle.7z.001', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.002', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.003', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.004', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.005', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.006', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.007', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.008', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.009', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.010', 5368709120),
    ('jpegai_train-natural-full_bundle.7z.011', 3978987972),
    ('jpegai_valid-natural-full_bundle.7z.001', 4041065075),
    ('jpegai_valid-natural_cropped_bundle.7z.001', 4038870070),
)

USER_AGENT = 'jpeg-ai-reference-software dataset downloader'
CHUNK = 1 << 20
# Rough size of the unpacked content relative to the archives, for the disk space check.
EXTRACTED_SIZE_FACTOR = 1.2

# The published names say what an archive holds: jpegai_train-natural-full_bundle.7z.001,
# jpegai_train-natural-cropped_00000-01299_bundle.7z.001, jpegai_train-extra_HF2000.tar,
# jpegai_valid-natural_cropped_bundle.7z.001.  Separators vary between - and _, so a name is
# normalised before it is read.
EXTRA_RE = re.compile(r'extra-([a-z0-9]+)', re.IGNORECASE)
RANGE_RE = re.compile(r'(\d{5}-\d{5})')
# Older names, from before the datasets moved to the ISO and ITU mirrors.
EXTRA_TOKENS = ('scc', 'hf2000', 'phfa500', 'phf200', 'lq7000', 'md300', 'excel300', 'cp50',
                'imscc')
PATCH_TOKENS = ('crop', 'patch')
TEST_TOKENS = ('test', )


class DatasetError(Exception):
    """A user-facing error: reported as a message, never as a traceback."""


# ######################################################################################################################
#  What an archive holds
# ######################################################################################################################
#  kind:
#     natural_full     full-size natural training images
#     natural_patches  natural training images cropped into patches
#     extra            an additional training dataset (screen content, high frequency, ...)
#     validation       the validation set
#     test             the test set, which the codec pulls with dvc rather than from here
#     unknown          could not be classified; offered separately, never selected by default
class RemoteFile:
    """One file on the mirror: a whole archive, or one volume of a split one."""

    def __init__(self, name, url, size=None):
        self.name = name
        self.url = url
        self.size = size
        self.volume = 0

    def __repr__(self):
        return f'RemoteFile({self.name}, {self.size})'


def normalised_name(name):
    """Strip the archive suffix and unify the separators, so one pattern reads every name."""
    stem = os.path.basename(name).lower()
    stem = re.sub(r'\.(7z|zip|tar|tgz)(\.\w+)*$', '', stem)
    return re.sub(r'[\s_]+', '-', stem)


def classify(name):
    """Work out what an archive holds from its name.

    Returns ``(kind, group, variant)``.  ``group`` labels the archive in the questionnaire --
    the name of an extra dataset, or the image range of a bundle -- and ``variant`` separates
    the full-size and cropped forms of the validation set.
    """
    stem = normalised_name(name)
    image_range = RANGE_RE.search(stem)
    image_range = image_range.group(1) if image_range else None

    if 'reference-software' in stem:
        return 'software', 'reference software', None
    if 'valid' in stem:
        # The validation set is published in two forms; a name that says neither is the one
        # single set the older layout had.
        if 'crop' in stem or 'patch' in stem:
            variant = 'cropped'
        elif 'full' in stem:
            variant = 'full'
        else:
            variant = None
        label = 'validation set' if variant is None else f'validation set, {variant}'
        return 'validation', label, variant
    if 'extra' in stem:
        match = EXTRA_RE.search(name.replace('_', '-'))
        return 'extra', match.group(1) if match else stem, None
    # Older names, from before the move to the ISO and ITU mirrors, name the extra dataset
    # without saying "extra" -- and scc7000_patchs2.tar also says "patch", so this has to come
    # before the cropped-content test below.
    for token in EXTRA_TOKENS:
        if token in stem:
            return 'extra', token, None
    for token in TEST_TOKENS:
        if token in stem:
            return 'test', 'test set', None
    if 'natural-full' in stem or ('full' in stem and 'crop' not in stem):
        return 'natural_full', image_range or 'full-size images', None
    if any(token in stem for token in PATCH_TOKENS):
        return 'natural_patches', image_range or 'cropped patches', None
    if 'train' in stem:
        return 'natural_full', image_range or 'full-size images', None
    return 'unknown', 'unclassified', None


def split_volume(name):
    """Split ``<base>.7z.003`` into ``('<base>.7z', 3)``; a plain archive keeps index 0."""
    match = VOLUME_RE.match(name)
    if match is None:
        return name, 0
    return match.group('base'), int(match.group('index'))


def is_archive(name):
    return split_volume(name)[0].lower().endswith(ARCHIVE_SUFFIXES)


class ArchiveSet:
    """One logical archive: a single file, or the volumes a large one is split into."""

    def __init__(self, name, kind, group, variant=None):
        self.name = name
        self.kind = kind
        self.group = group
        self.variant = variant
        self.volumes = list()

    @property
    def size(self):
        if any(x.size is None for x in self.volumes):
            return None
        return sum(x.size for x in self.volumes)

    @property
    def is_split(self):
        return len(self.volumes) > 1 or self.volumes[0].name != self.name

    @property
    def first_volume(self):
        return self.volumes[0]

    def label(self):
        count = len(self.volumes)
        if not self.is_split:
            return self.name
        return f'{self.name} ({count} volume{"" if count == 1 else "s"})'

    def __repr__(self):
        return f'ArchiveSet({self.name}, {self.kind}, {len(self.volumes)} volume(s))'


def group_volumes(files):
    """Gather the volumes of each split archive into one ArchiveSet, keeping listing order."""
    sets = dict()
    for remote in files:
        base, index = split_volume(remote.name)
        archive = sets.get(base)
        if archive is None:
            kind, group, variant = classify(base)
            archive = ArchiveSet(base, kind, group, variant)
            sets[base] = archive
        remote.volume = index
        archive.volumes.append(remote)
    for archive in sets.values():
        archive.volumes.sort(key=lambda x: (x.volume, x.name))
    return sorted(sets.values(), key=lambda x: x.name)


# ######################################################################################################################
#  Reading the published directory index
# ######################################################################################################################
SIZE_RE = re.compile(r'(?<![\w.])(\d+(?:[.,]\d+)?)\s*([KMGT]i?B?|B)(?![\w.])', re.IGNORECASE)
BYTES_RE = re.compile(r'(?<![\w.])(\d{4,})(?![\w.])')
SIZE_UNITS = {'b': 1, 'k': 1 << 10, 'm': 1 << 20, 'g': 1 << 30, 't': 1 << 40}
LINK_MARK = '\x00'
# Dates and times share the row with the size; removing them keeps a year out of the numbers.
DATE_RE = re.compile(r'\d{1,2}[-/]\w{3,9}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|'
                     r'\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}:\d{2}(?::\d{2})?')


def parse_size(text):
    """Read a file size out of the listing text around a link.

    A size written with a unit wins, and the last one on the line is taken, because a listing
    row reads "name date size".  Failing that the largest bare number of four digits or more is
    used: on a row such as "1/12/2026  3:04 PM  30064771072 name" that is the size rather than
    the year.  Anything smaller is left unknown for a HEAD request to fill in.
    """
    if not text:
        return None
    matches = SIZE_RE.findall(text)
    if matches:
        value, unit = matches[-1]
        return int(float(value.replace(',', '.')) * SIZE_UNITS.get(unit[0].lower(), 1))
    numbers = BYTES_RE.findall(DATE_RE.sub(' ', text))
    if numbers:
        return max(int(x) for x in numbers)
    return None


class IndexParser(html.parser.HTMLParser):
    """Flatten a directory index into rows, each holding its links and its text.

    Directory listings put the size of a file on the same row as its link, but on which side
    depends on the server: an Apache listing writes "name date size", an IIS one writes
    "date size name".  Rebuilding the rows -- physical lines in a ``<pre>`` listing, ``<tr>``
    elements in a table -- keeps each size with the file it belongs to either way, which
    picking the text before or after a link cannot do.
    """

    ROW_BREAK_TAGS = ('br', 'tr', 'p', 'li', 'hr', 'div', 'h1', 'h2', 'table', 'pre', 'ul')
    CELL_TAGS = ('td', 'th')

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hrefs = list()
        self._chunks = list()
        self._in_table = 0
        self._href = None

    # -- collecting ---------------------------------------------------------------------------
    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == 'table':
            self._in_table += 1
        if tag == 'a':
            for key, value in attrs:
                if key.lower() == 'href':
                    self._href = value
                    return
            return
        self._break_or_space(tag)

    def handle_startendtag(self, tag, attrs):
        self._break_or_space(tag.lower())

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == 'table':
            self._in_table = max(0, self._in_table - 1)
        if tag == 'a':
            if self._href is not None:
                self._chunks.append(f'{LINK_MARK}{len(self.hrefs)}{LINK_MARK}')
                self.hrefs.append(self._href)
                self._href = None
            return
        self._break_or_space(tag)

    def handle_data(self, data):
        if self._href is not None:
            # The link text itself never carries the size; keep the row readable without it.
            return
        self._chunks.append(data.replace('\n', ' ') if self._in_table else data)

    def _break_or_space(self, tag):
        if tag in self.ROW_BREAK_TAGS:
            self._chunks.append('\n')
        elif tag in self.CELL_TAGS:
            self._chunks.append(' ')

    # -- results ------------------------------------------------------------------------------
    def rows(self):
        """Yield ``(href, size)`` for every link, with the size found on its own row."""
        text = ''.join(self._chunks)
        for line in text.split('\n'):
            indexes = [int(x) for x in re.findall(f'{LINK_MARK}(\\d+){LINK_MARK}', line)]
            if not indexes:
                continue
            bare = re.sub(f'{LINK_MARK}\\d+{LINK_MARK}', ' ', line)
            # A row with several links cannot say which one a size belongs to.
            size = parse_size(bare) if len(indexes) == 1 else None
            for index in indexes:
                yield self.hrefs[index], size


def parse_index(html_text):
    """Return ``[(href, size)]`` for a directory index page."""
    parser = IndexParser()
    parser.feed(html_text)
    return list(parser.rows())

# ######################################################################################################################
#  HTTP
# ######################################################################################################################
def http_open(url, headers=None, method=None, timeout=60):
    request = urllib.request.Request(url, method=method or 'GET')
    request.add_header('User-Agent', USER_AGENT)
    for key, value in (headers or dict()).items():
        request.add_header(key, value)
    return urllib.request.urlopen(request, timeout=timeout)


def fetch_text(url, timeout=60):
    try:
        with http_open(url, timeout=timeout) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or 'utf-8'
    except urllib.error.HTTPError as error:
        raise DatasetError(f'{url} answered {error.code} {error.reason}')
    except (urllib.error.URLError, OSError) as error:
        raise DatasetError(f'Could not reach {url}: {error}')
    return raw.decode(charset, 'replace')


def head_probe(url, timeout=30):
    """Ask the server about a file: returns ``(state, size)``.

    ``state`` is 'ok' when the file is there, 'missing' when the server says it is not, and
    'unknown' when the question could not be answered.
    """
    try:
        with http_open(url, method='HEAD', timeout=timeout) as response:
            length = response.headers.get('Content-Length')
            return 'ok', int(length) if length is not None else None
    except urllib.error.HTTPError as error:
        return ('missing', None) if error.code in (403, 404, 410) else ('unknown', None)
    except (urllib.error.URLError, OSError, ValueError):
        return 'unknown', None


def head_size(url, timeout=30):
    """Ask the server how large a file is; returns None when it will not say."""
    return head_probe(url, timeout=timeout)[1]


def catalogue_from_known(base_url, use_head=True):
    """Build the file list from the published catalogue, for a mirror with no index.

    The names are the ones both mirrors publish; each is confirmed against the server, so a
    catalogue that has moved on does not invent files that are not there.
    """
    ans = list()
    for name, size in KNOWN_CATALOGUE:
        url = urllib.parse.urljoin(base_url, name)
        if not use_head:
            ans.append(RemoteFile(name, url, size))
            continue
        state, exact = head_probe(url)
        if state == 'missing':
            continue
        ans.append(RemoteFile(name, url, exact if exact is not None else size))
    return ans


def crawl(base_url, depth=2, use_head=True, verbose=False):
    """Walk the published index of ``base_url`` and return its archives and checksum files."""
    base_url = base_url if base_url.endswith('/') else base_url + '/'
    files, checksums, visited = dict(), dict(), set()
    queue = [(base_url, 0)]

    while queue:
        url, level = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        if verbose:
            print(f'  reading {url}')
        try:
            page = fetch_text(url)
        except DatasetError as error:
            if url == base_url:
                raise
            # A subdirectory that cannot be read is worth a word, not a failed run.
            print(f'  warning: skipping {url}: {error}')
            continue
        for href, size in parse_index(page):
            absolute = urllib.parse.urljoin(url, href)
            absolute, _ = urllib.parse.urldefrag(absolute)
            if not absolute.startswith(base_url) or absolute in (url, base_url):
                continue
            if absolute.endswith('/') and url.startswith(absolute):
                continue        # a link back up the tree, such as "[To Parent Directory]"
            name = urllib.parse.unquote(
                os.path.basename(urllib.parse.urlparse(absolute).path.rstrip('/')))
            if not name:
                continue
            if absolute.endswith('/'):
                if level < depth:
                    queue.append((absolute, level + 1))
                continue
            if name.lower().endswith(CHECKSUM_SUFFIXES):
                checksums.setdefault(os.path.splitext(name)[0], absolute)
                continue
            if not is_archive(name) or absolute in files:
                continue
            files[absolute] = RemoteFile(name, absolute, size)

    ans = sorted(files.values(), key=lambda x: x.name)
    if use_head:
        for remote in ans:
            exact = head_size(remote.url)
            if exact is not None:
                remote.size = exact
    return ans, checksums


# ######################################################################################################################
#  Formatting
# ######################################################################################################################
def format_size(num_bytes):
    if num_bytes is None:
        return 'unknown size'
    value = float(num_bytes)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024.0 or unit == 'TiB':
            return f'{int(value)} B' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1024.0
    return f'{value:.1f} TiB'


def total_size(files):
    """Total size of a selection; None as soon as one size is unknown."""
    if any(x.size is None for x in files):
        return None
    return sum(x.size for x in files)


def format_total(files):
    known = [x for x in files if x.size is not None]
    unknown = len(files) - len(known)
    if not known:
        return f'{len(files)} file(s), size unknown'
    text = format_size(sum(x.size for x in known))
    if unknown:
        text += f' plus {unknown} file(s) of unknown size'
    return f'{len(files)} file(s), {text}'


def format_total_sets(archives):
    """Same, counted in whole archives rather than in the volumes they are split into."""
    known = [x for x in archives if x.size is not None]
    unknown = len(archives) - len(known)
    if not known:
        return f'{len(archives)} archive(s), size unknown'
    text = format_size(sum(x.size for x in known))
    if unknown:
        text += f' plus {unknown} archive(s) of unknown size'
    return f'{len(archives)} archive(s), {text}'


def display_path(path):
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


# ######################################################################################################################
#  Asking
# ######################################################################################################################
def _input(prompt):
    if not sys.stdin.isatty():
        raise DatasetError('This run needs an answer but there is no terminal to ask on. '
                           'Pass the choices as options (see --help) or use --answers FILE.')
    try:
        return input(prompt)
    except EOFError:
        raise DatasetError('No answer given.')


def ask_choice(question, options, default=None):
    """Ask a single-choice question; ``options`` is a list of ``(key, label)`` pairs."""
    keys = [key for key, _ in options]
    default = default if default in keys else keys[0]
    print(f'\n{question}')
    for index, (key, label) in enumerate(options, start=1):
        marker = '*' if key == default else ' '
        print(f'  {index}){marker} {label}')
    while True:
        answer = _input(f'Choice [{keys.index(default) + 1}]: ').strip().lower()
        if not answer:
            return default
        if answer in keys:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return keys[int(answer) - 1]
        print('  Please answer with the number of one of the options.')


def ask_yes_no(question, default=True):
    suffix = '[Y/n]' if default else '[y/N]'
    while True:
        answer = _input(f'{question} {suffix}: ').strip().lower()
        if not answer:
            return default
        if answer in ('y', 'yes'):
            return True
        if answer in ('n', 'no'):
            return False
        print('  Please answer y or n.')


def ask_subset(question, options):
    """Ask which of ``(key, label)`` options are wanted: all, none or a chosen few."""
    print(f'\n{question}')
    for index, (key, label) in enumerate(options, start=1):
        print(f'  {index}) {label}')
    print('  a) all of them')
    print('  n) none of them')
    keys = [key for key, _ in options]
    while True:
        answer = _input('Choice [n]: ').strip().lower()
        if answer in ('', 'n', 'none', 'no'):
            return list()
        if answer in ('a', 'all'):
            return list(keys)
        picked, bad = list(), False
        for token in re.split(r'[\s,]+', answer):
            if not token:
                continue
            if token.isdigit() and 1 <= int(token) <= len(options):
                picked.append(keys[int(token) - 1])
            elif token in keys:
                picked.append(token)
            else:
                bad = True
        if picked and not bad:
            return sorted(set(picked), key=keys.index)
        print('  Please answer with numbers, "a" for all or "n" for none.')


# ######################################################################################################################
#  Choosing what to download
# ######################################################################################################################
def group_by_kind(archives):
    ans = dict()
    for archive in archives:
        ans.setdefault(archive.kind, list()).append(archive)
    return ans


def group_by_label(archives):
    """Archives keyed by the dataset they belong to, keeping the order they were listed in."""
    ans = dict()
    for archive in archives:
        ans.setdefault(archive.group, list()).append(archive)
    return ans


def describe_group(archives):
    """How much a dataset adds to the download, and over how many archives and volumes."""
    volumes = sum(len(x.volumes) for x in archives)
    if len(archives) == 1:
        text = format_size(archives[0].size)
    else:
        text = format_total_sets(archives)
    if volumes > len(archives):
        text += f' in {volumes} volumes'
    return text


def select_files(args, archives, interactive):
    """Work through the questions and return the archives to download."""
    by_kind = group_by_kind(archives)
    selection = list()

    options = [('train', 'training set'), ('validation', 'validation set'),
               ('both', 'both of them')]
    datasets = args.datasets
    if datasets is None:
        datasets = ask_choice('Which datasets do you need?', options,
                              default='both') if interactive else 'both'
    want_train = datasets in ('train', 'both')
    want_valid = datasets in ('validation', 'both')

    if want_train:
        full = by_kind.get('natural_full', list())
        patches = by_kind.get('natural_patches', list())
        natural_options = list()
        if full:
            natural_options.append(('full', f'full-size images -- {describe_group(full)}'))
        if patches:
            natural_options.append(('patches', f'cropped patches -- {describe_group(patches)}'))
        natural_options.append(('none', 'skip the natural content'))

        natural = args.natural
        if natural is None:
            if not full and not patches:
                print('\nThe mirror offers no natural training content.')
                natural = 'none'
            else:
                default = 'patches' if patches else 'full'
                natural = ask_choice('Natural training content -- which form?',
                                     natural_options, default=default) if interactive \
                    else default
        if natural == 'full':
            selection += full
        elif natural == 'patches':
            selection += patches

        extras = by_kind.get('extra', list())
        if extras:
            groups = group_by_label(extras)
            group_options = [(name.lower(), f'{name} -- {describe_group(group)}')
                             for name, group in sorted(groups.items())]
            by_key = {name.lower(): group for name, group in groups.items()}
            wanted = args.extras
            if wanted is None:
                wanted = ask_subset(
                    f'Extra training datasets ({describe_group(extras)} in total) -- '
                    'which ones?', group_options) if interactive else list()
            elif wanted == ['all']:
                wanted = sorted(by_key)
            elif wanted == ['none']:
                wanted = list()
            wanted = [x.lower() for x in wanted]
            unknown = [x for x in wanted if x not in by_key]
            if unknown:
                raise DatasetError('Unknown extra dataset(s): {}. Available: {}'.format(
                    ', '.join(unknown), ', '.join(sorted(by_key)) or 'none'))
            for name in wanted:
                selection += by_key[name]

    if want_valid:
        selection += select_validation(args, by_kind.get('validation', list()), interactive)

    software = by_kind.get('software', list())
    if software:
        print('\nnote: the mirror also carries the reference software itself ({}); '
              'this script only downloads datasets.'.format(describe_group(software)))

    leftovers = by_kind.get('unknown', list())
    if leftovers and interactive and args.extras is None:
        print(f'\n{len(leftovers)} archive(s) could not be placed by name.')
        picked = ask_subset('Download any of them?',
                            [(x.name, f'{x.label()} -- {format_size(x.size)}')
                             for x in leftovers])
        selection += [x for x in leftovers if x.name in picked]
    elif leftovers:
        print(f'\nnote: ignoring {len(leftovers)} archive(s) that could not be placed; '
              'see --list-remote')

    # Keep the listing order and drop anything selected twice.
    seen, ans = set(), list()
    for archive in selection:
        if archive.name not in seen:
            seen.add(archive.name)
            ans.append(archive)
    return ans


def select_validation(args, validation, interactive):
    """Question 4: show what the validation set costs, in each form it is published in."""
    if not validation:
        print('\nThe mirror offers no validation set.')
        return list()

    variants = group_by_label(validation)
    print(f'\nValidation set: {describe_group(validation)}')
    for name, group in sorted(variants.items()):
        print(f'  {name} -- {describe_group(group)}')

    if args.validation is not None:
        wanted = args.validation
    elif not interactive or args.yes:
        wanted = 'cropped' if any(x.variant == 'cropped' for x in validation) else 'all'
    elif len(variants) < 2:
        return validation if ask_yes_no('Download the validation set?', default=True) \
            else list()
    else:
        options = list()
        for variant in ('cropped', 'full'):
            group = [x for x in validation if x.variant == variant]
            if group:
                options.append((variant, f'{variant} -- {describe_group(group)}'))
        options.append(('all', f'both forms -- {describe_group(validation)}'))
        options.append(('none', 'skip the validation set'))
        wanted = ask_choice('Which form of the validation set?', options, default='cropped')

    if wanted == 'none':
        return list()
    if wanted == 'all':
        return validation
    chosen = [x for x in validation if x.variant == wanted]
    if not chosen:
        raise DatasetError(f'The mirror has no "{wanted}" validation set; it offers: '
                           + ', '.join(sorted({str(x.variant) for x in validation})))
    return chosen


def confirm_plan(args, selection, states, policy, data_dir, interactive):
    """Show the summary -- what, how big, where -- and ask to go ahead."""
    by_name = {x.archive.name: x for x in states}
    rows = [[x.label(), x.kind, format_size(x.size),
             by_name[x.name].state if x.name in by_name else '-'] for x in selection]
    print('\nAbout to download:')
    print(format_table(['archive', 'content', 'size', 'on disk'], rows))

    download = bytes_for_policy(states, policy)
    content = total_size(selection)
    print(f'\nTotal download: {format_size(download)}')
    if download != content:
        print(f'Chosen datasets: {format_total_sets(selection)}, '
              'the rest is already on disk')
    if args.unpack:
        extracted = None if content is None else int(content * EXTRACTED_SIZE_FACTOR)
        needed = None if download is None or extracted is None else (
            max(download, extracted + max([x.size or 0 for x in selection], default=0))
            if args.remove_archives else download + extracted)
        print(f'Unpacked content adds about {format_size(extracted)}, '
              f'so about {format_size(needed)} of free disk is needed')
    else:
        needed = download
    print(f'Destination: {display_path(data_dir)}')
    if os.path.isdir(data_dir):
        free = shutil.disk_usage(data_dir).free
        print(f'Free space there: {format_size(free)}')
        if needed is not None and free < needed and not args.no_space_check:
            raise DatasetError(
                f'Not enough free space: {format_size(free)} available, about '
                f'{format_size(needed)} needed. Choose fewer datasets, point --data-dir at a '
                'larger filesystem, or pass --no-space-check.')

    if args.yes or not interactive:
        return True
    return ask_yes_no('\nStart the download?', default=True)


# ######################################################################################################################
#  What is already on disk
# ######################################################################################################################
# A receipt in the data directory remembers which archive produced which files, so a later run
# can tell "downloaded" from "downloaded and unpacked" without guessing from what the dataset
# directories happen to contain.
RECEIPT_NAME = '.jpegai_datasets.json'


def load_receipt(data_dir):
    path = os.path.join(data_dir, RECEIPT_NAME)
    if not os.path.isfile(path):
        return dict()
    try:
        with open(path, 'r') as f:
            receipt = json.load(f)
    except (ValueError, OSError):
        return dict()
    archives = receipt.get('archives')
    return archives if isinstance(archives, dict) else dict()


def save_receipt(data_dir, archives):
    with open(os.path.join(data_dir, RECEIPT_NAME), 'w') as f:
        json.dump({'archives': archives}, f, indent=4, sort_keys=True)
        f.write('\n')


class LocalState:
    """How one archive on the mirror compares with what this machine already has."""

    def __init__(self, archive, archives_dir, unpacked):
        self.archive = archive
        self.unpacked = unpacked        # receipt entry, or None
        self.volumes = list()
        for remote in archive.volumes:
            path = os.path.join(archives_dir, remote.name)
            size = os.path.getsize(path) if os.path.isfile(path) else None
            self.volumes.append((remote, path, size))

    @property
    def local_size(self):
        sizes = [size for _, _, size in self.volumes if size is not None]
        return sum(sizes) if sizes else None

    @property
    def state(self):
        states = [volume_state(remote, size) for remote, _, size in self.volumes]
        for name in ('oversized', 'partial', 'missing', 'present'):
            if name in states:
                # A split archive is only complete when every volume is.
                return 'partial' if (name == 'missing' and any(
                    x != 'missing' for x in states)) else name
        return 'complete'

    @property
    def is_complete(self):
        return self.state in ('complete', 'present')

    @property
    def unpacked_matches(self):
        """True when the unpacked content came from the archive the mirror offers now."""
        if not self.unpacked:
            return False
        recorded = self.unpacked.get('size')
        return self.archive.size is None or recorded is None or recorded == self.archive.size

    def missing_bytes(self):
        """Bytes still to fetch to finish this archive; None if a size is unknown."""
        total = 0
        for remote, _, size in self.volumes:
            if remote.size is None:
                return None
            if size is None:
                total += remote.size
            elif size < remote.size:
                total += remote.size - size
            elif size > remote.size:
                total += remote.size
        return total

    def describe_unpacked(self):
        if not self.unpacked:
            return 'no'
        when = str(self.unpacked.get('unpacked_at', ''))[:10]
        text = '{} file(s)'.format(self.unpacked.get('files', '?'))
        if when:
            text += f' on {when}'
        return text if self.unpacked_matches else f'{text}, from an older archive'


def volume_state(remote, local_size):
    if local_size is None:
        return 'missing'
    if remote.size is None:
        return 'present'
    if local_size < remote.size:
        return 'partial'
    if local_size > remote.size:
        return 'oversized'
    return 'complete'


def inspect_local(archives, archives_dir, receipt):
    """Compare every chosen archive with the copy and the unpacked content on disk."""
    return [LocalState(x, archives_dir, receipt.get(x.name)) for x in archives]


def print_existing(states):
    rows = [[x.archive.label(),
             format_size(x.local_size) if x.local_size is not None else '-',
             format_size(x.archive.size), x.state, x.describe_unpacked()] for x in states]
    print(format_table(['archive', 'on disk', 'on mirror', 'state', 'unpacked'], rows))


def bytes_for_policy(states, policy):
    """How much a policy actually downloads; None as soon as one size is unknown."""
    total = 0
    for state in states:
        if policy == 'redownload':
            needed = state.archive.size
        elif policy == 'skip':
            needed = state.archive.size if state.state == 'missing' else 0
        else:
            needed = state.missing_bytes()
        if needed is None:
            return None
        total += needed
    return total


def ask_existing_policy(args, states, interactive):
    """Something is already here: finish it, fetch it again, or leave it alone?"""
    present = [x for x in states if x.state != 'missing']
    if not present:
        return args.existing or 'resume'

    print('\nSome of what was chosen is already on disk:')
    print_existing(present)

    partial = [x for x in present if x.state in ('partial', 'oversized')]
    unpacked = [x for x in present if x.unpacked]
    for state in partial:
        print(f'  {state.archive.name} is incomplete, {format_size(state.missing_bytes())} '
              'still to download')
    if unpacked:
        print(f'  {len(unpacked)} archive(s) have already been unpacked')

    if args.existing is not None:
        return args.existing
    if not interactive:
        return 'resume'

    options = [
        ('resume', 'finish what is incomplete, keep what is complete -- downloads {}'.format(
            format_size(bytes_for_policy(states, 'resume')))),
        ('redownload', 'download all of it again -- downloads {}'.format(
            format_size(bytes_for_policy(states, 'redownload')))),
        ('skip', 'leave what is on disk alone, fetch only what is missing entirely -- '
                 'downloads {}'.format(format_size(bytes_for_policy(states, 'skip')))),
    ]
    return ask_choice('What should be done with what is already there?', options,
                      default='resume')


def ask_reunpack(args, states, interactive):
    """Archives whose content is already unpacked: unpack them again or leave them?"""
    done = [x for x in states if x.unpacked and x.unpacked_matches]
    if not done or not args.unpack:
        return args.reunpack if args.reunpack is not None else False
    if args.reunpack is not None:
        return args.reunpack
    if not interactive:
        return False
    print(f'\n{len(done)} of the chosen archive(s) have already been unpacked:')
    for state in done:
        into = state.unpacked.get('into', '?')
        print(f'  {state.archive.name} -> {into} ({state.describe_unpacked()})')
    return ask_yes_no('Unpack them again?', default=False)


# ######################################################################################################################
#  Downloading
# ######################################################################################################################
def download_file(remote, dest_dir, retries=3, quiet=False, policy='resume'):
    """Fetch one file, following the policy for a copy that is already there.

    ``resume`` finishes a partial file with a Range request and leaves a complete one alone,
    ``redownload`` fetches it again from the start, and ``skip`` keeps whatever is on disk.
    """
    path = os.path.join(dest_dir, remote.name)
    have = os.path.getsize(path) if os.path.isfile(path) else 0

    if have and policy == 'skip':
        print(f'  {remote.name}: left as it is ({format_size(have)})')
        return 'skipped'
    if have and policy == 'redownload':
        os.remove(path)
        have = 0

    if remote.size is not None:
        if have == remote.size:
            print(f'  {remote.name}: already complete ({format_size(have)})')
            return 'complete'
        if have > remote.size:
            print(f'  {remote.name}: local copy is larger than the one on the mirror, '
                  'downloading it again')
            os.remove(path)
            have = 0
        elif have:
            print(f'  {remote.name}: resuming, {format_size(remote.size - have)} to go')

    last_error = None
    for attempt in range(1, retries + 1):
        headers = {'Range': f'bytes={have}-'} if have else dict()
        try:
            with http_open(remote.url, headers=headers) as response:
                partial = response.status == 206
                if have and not partial:
                    # The server ignored the range, so start over rather than corrupt the file.
                    have = 0
                total = response.headers.get('Content-Length')
                total = (int(total) + have) if total is not None else remote.size
                mode = 'ab' if have else 'wb'
                started, done = time.time(), have
                with open(path, mode) as f:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if not quiet:
                            report_progress(remote.name, done, total, started)
            if not quiet:
                report_progress(remote.name, done, total, started, final=True)
            if remote.size is None or os.path.getsize(path) == remote.size:
                return 'resumed' if headers else 'fresh'
            have = os.path.getsize(path)
            last_error = 'incomplete transfer'
        except urllib.error.HTTPError as error:
            if error.code == 416 and remote.size is None:
                # Range past the end: the file is already fully downloaded.
                return 'complete'
            last_error = f'{error.code} {error.reason}'
            have = os.path.getsize(path) if os.path.isfile(path) else 0
        except (urllib.error.URLError, OSError) as error:
            last_error = str(error)
            have = os.path.getsize(path) if os.path.isfile(path) else 0
        if attempt < retries:
            delay = 2 ** attempt
            print(f'  {remote.name}: {last_error}; resuming in {delay}s')
            time.sleep(delay)
    raise DatasetError(f'Could not download {remote.name}: {last_error}. '
                       'Run the script again to resume.')


def report_progress(name, done, total, started, final=False):
    elapsed = max(time.time() - started, 1e-6)
    speed = format_size(int(done / elapsed))
    if total:
        percent = 100.0 * done / total
        text = f'  {name}: {percent:5.1f}% of {format_size(total)} at {speed}/s'
    else:
        text = f'  {name}: {format_size(done)} at {speed}/s'
    if sys.stdout.isatty():
        sys.stdout.write('\r' + text + (' \n' if final else '   '))
    elif final:
        sys.stdout.write(text + '\n')
    sys.stdout.flush()


def verify_checksum(path, checksum_url, quiet=False):
    """Check a downloaded archive against a published checksum; None when there is none."""
    algorithm = os.path.splitext(checksum_url)[1].lstrip('.').lower()
    if algorithm not in ('md5', 'sha1', 'sha256'):
        return None
    try:
        published = fetch_text(checksum_url, timeout=30).split()
    except DatasetError:
        return None
    if not published:
        return None
    expected = published[0].strip().lower()
    digest = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if not quiet:
        print(f'  {os.path.basename(path)}: {algorithm} '
              f'{"ok" if actual == expected else "MISMATCH"}')
    return actual == expected


# ######################################################################################################################
#  Unpacking
# ######################################################################################################################
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.yuv')


class ExtractPlan:
    """Where an archive's content goes and which of its members are kept."""

    def __init__(self, directory, flatten, keep):
        self.directory = directory
        self.flatten = flatten
        self.keep = keep


def plan_for(archive):
    if archive.kind == 'natural_full':
        return ExtractPlan(TRAIN_FULL_DIR, True, IMAGE_SUFFIXES)
    if archive.kind == 'natural_patches':
        return ExtractPlan(TRAIN_CROP_DIR, True, IMAGE_SUFFIXES)
    if archive.kind == 'extra':
        # Keep the archive's own directories: the training lists in cfg/training_list refer to
        # the extra datasets by directory, for example jpegai_training_7000scc/<image>.png.
        return ExtractPlan(TRAIN_CROP_DIR, False, IMAGE_SUFFIXES)
    if archive.kind == 'validation':
        # The cropped form is the one scripts/train.sh reads; the full-size one is kept beside
        # it.  Both may ship their own file list next to the images.
        directory = VALID_FULL_DIR if archive.variant == 'full' else VALID_DIR
        return ExtractPlan(directory, True, IMAGE_SUFFIXES + ('.txt', ))
    stem = re.sub(r'\.(zip|tar|7z|tgz|tar\.gz|tar\.bz2|tar\.xz)$', '', archive.name,
                  flags=re.I)
    return ExtractPlan(stem, False, IMAGE_SUFFIXES + ('.txt', ))


def safe_member_path(name, strip=0):
    """Reject anything that would write outside the destination, and strip leading levels."""
    name = name.replace('\\', '/').lstrip('/')
    parts = [x for x in name.split('/') if x not in ('', '.')]
    if any(x == '..' for x in parts):
        return None
    parts = parts[strip:]
    return '/'.join(parts) if parts else None


def sevenzip_tool():
    """The 7-Zip command line, under any of the names it is installed as."""
    for name in ('7z', '7za', '7zz', '7zr'):
        if shutil.which(name) is not None:
            return name
    return None


def require_sevenzip(archives):
    """7-Zip bundles cannot be unpacked with the standard library; check before downloading."""
    if not any(x.name.lower().endswith('.7z') for x in archives):
        return
    if sevenzip_tool() is None:
        raise DatasetError(
            'The datasets are published as 7-Zip bundles and no 7z command was found. '
            'Install it with "sudo apt install p7zip-full", or pass --no-unpack to download '
            'the archives now and unpack them later.')


def count_images(root):
    total = 0
    for _, _, files in os.walk(root):
        total += sum(1 for x in files if x.lower().endswith(IMAGE_SUFFIXES))
    return total


def extract_with_sevenzip(archive_path, dest_dir, plan, quiet=False):
    """Unpack a 7-Zip archive, following on to its later volumes on its own."""
    tool = sevenzip_tool()
    if tool is None:
        raise DatasetError('No 7z command found; install it with '
                           '"sudo apt install p7zip-full".')
    before = count_images(dest_dir)
    # "e" drops the paths, "x" keeps them; the masks keep everything else out of the dataset.
    argv = [tool, 'e' if plan.flatten else 'x', '-y', '-r', f'-o{dest_dir}', archive_path]
    argv += [f'*{suffix}' for suffix in plan.keep]
    process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
    if process.returncode != 0:
        raise DatasetError('{} failed on {}:\n{}'.format(
            tool, os.path.basename(archive_path), process.stdout.strip()[-2000:]))
    written = count_images(dest_dir) - before
    roots = set(os.listdir(dest_dir)) if not plan.flatten else set()
    if not quiet:
        print(f'  {os.path.basename(archive_path)} -> {display_path(dest_dir)}: '
              f'{written} file(s)')
    return written, roots


def extract_archive(archive_path, dest_dir, plan, strip=0, quiet=False):
    """Unpack the wanted members of one archive; returns (files written, top-level names)."""
    os.makedirs(dest_dir, exist_ok=True)
    base = split_volume(os.path.basename(archive_path))[0].lower()
    if base.endswith('.7z'):
        return extract_with_sevenzip(archive_path, dest_dir, plan, quiet=quiet)

    written, roots = 0, set()

    def target_for(member_name):
        relative = safe_member_path(member_name, strip=strip)
        if relative is None:
            return None
        if not relative.lower().endswith(plan.keep):
            return None
        relative = os.path.basename(relative) if plan.flatten else relative
        roots.add(relative.split('/')[0])
        return os.path.join(dest_dir, *relative.split('/'))

    if base.endswith('.zip'):
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                target = target_for(info.filename)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(info) as source, open(target, 'wb') as sink:
                    shutil.copyfileobj(source, sink, CHUNK)
                written += 1
    else:
        with tarfile.open(archive_path) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                target = target_for(member.name)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    continue
                with source, open(target, 'wb') as sink:
                    shutil.copyfileobj(source, sink, CHUNK)
                written += 1

    if not quiet:
        print(f'  {os.path.basename(archive_path)} -> {display_path(dest_dir)}: '
              f'{written} file(s)')
    return written, roots


# ######################################################################################################################
#  File lists
# ######################################################################################################################
def collect_images(root):
    """Every image under ``root``, as sorted paths relative to it."""
    ans = list()
    for dir_path, _, files in os.walk(root):
        rel_dir = os.path.relpath(dir_path, root)
        for name in files:
            if not name.lower().endswith(IMAGE_SUFFIXES):
                continue
            relative = name if rel_dir == '.' else os.path.join(rel_dir, name)
            ans.append(relative.replace(os.sep, '/'))
    return sorted(ans)


def write_list_file(path, names):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for name in names:
            f.write(f'{name}\n')


def generate_lists(data_dir, touched_dirs, force=False):
    """Write the file lists the training scripts read, for the directories that changed."""
    made = list()
    for directory, list_name in ((TRAIN_CROP_DIR, TRAIN_LIST), (VALID_DIR, VALID_LIST),
                                (VALID_FULL_DIR, VALID_LIST)):
        if directory not in touched_dirs:
            continue
        root = os.path.join(data_dir, directory)
        if not os.path.isdir(root):
            continue
        list_path = os.path.join(root, list_name)
        # The validation archive ships its own list; only step in when it is absent.
        if list_name == VALID_LIST and os.path.isfile(list_path) and not force:
            continue
        names = collect_images(root)
        write_list_file(list_path, names)
        made.append((list_path, len(names)))
    return made


# ######################################################################################################################
#  Status
# ######################################################################################################################
def print_status(data_dir, archives_dir):
    """Report what is on disk: unpacked datasets, archives, and what came from where."""
    rows = list()
    for directory, list_name in ((TRAIN_FULL_DIR, None), (TRAIN_CROP_DIR, TRAIN_LIST),
                                 (VALID_DIR, VALID_LIST), (VALID_FULL_DIR, VALID_LIST)):
        root = os.path.join(data_dir, directory)
        if not os.path.isdir(root):
            rows.append([directory, 'absent', '-'])
            continue
        listed = '-'
        if list_name is not None and os.path.isfile(os.path.join(root, list_name)):
            with open(os.path.join(root, list_name), 'r') as f:
                listed = str(sum(1 for line in f if line.strip()))
        rows.append([directory, str(count_images(root)), listed])
    print(f'Datasets in {display_path(data_dir)}:')
    print(format_table(['directory', 'images', 'entries in list'], rows))

    receipt = load_receipt(data_dir)
    if receipt:
        rows = [[name, entry.get('into', '?'), str(entry.get('files', '?')),
                 str(entry.get('unpacked_at', ''))[:10], format_size(entry.get('size'))]
                for name, entry in sorted(receipt.items())]
        print('\nUnpacked from:')
        print(format_table(['archive', 'into', 'files', 'when', 'archive size'], rows))

    if os.path.isdir(archives_dir):
        names = sorted(x for x in os.listdir(archives_dir) if is_archive(x))
        if names:
            rows = list()
            for base, volumes in sorted(group_local_volumes(names).items()):
                size = sum(os.path.getsize(os.path.join(archives_dir, x)) for x in volumes)
                label = base if len(volumes) == 1 else f'{base} ({len(volumes)} volumes)'
                rows.append([label, classify(base)[0], format_size(size)])
            print(f'\nArchives in {display_path(archives_dir)}:')
            print(format_table(['archive', 'content', 'size'], rows))


def group_local_volumes(names):
    """Group the file names of split archives back into one entry per archive."""
    ans = dict()
    for name in names:
        ans.setdefault(split_volume(name)[0], list()).append(name)
    return ans


# ######################################################################################################################
#  Command line
# ######################################################################################################################
ANSWER_KEYS = ('source', 'base_url', 'datasets', 'natural', 'validation', 'extras',
               'unpack', 'remove_archives', 'existing', 'reunpack', 'data_dir',
               'archives_dir')


def parse_extras(text):
    lowered = text.strip().lower()
    if lowered in ('all', 'none'):
        return [lowered]
    return [x.strip() for x in text.split(',') if x.strip()]


def load_answers(path, args):
    """Fill in unanswered options from a saved answers file."""
    with open(path, 'r') as f:
        answers = json.load(f)
    unknown = [x for x in answers if x not in ANSWER_KEYS]
    if unknown:
        raise DatasetError('Unknown key(s) in {}: {}. Known keys: {}'.format(
            path, ', '.join(sorted(unknown)), ', '.join(ANSWER_KEYS)))
    for key in ANSWER_KEYS:
        if key in answers and getattr(args, key, None) in (None, ):
            setattr(args, key, answers[key])
    return args


def save_answers(path, args, source):
    answers = {key: getattr(args, key, None) for key in ANSWER_KEYS}
    # A custom --base-url is stored on its own; "custom" is not a source the parser accepts.
    answers['source'] = source if source in MIRRORS else None
    with open(path, 'w') as f:
        json.dump({k: v for k, v in answers.items() if v is not None}, f, indent=4,
                  sort_keys=True)
        f.write('\n')
    print(f'Answers saved to {display_path(path)}; re-run with --answers {display_path(path)}')


def resolve_source(args, interactive):
    """Question 1: which mirror to download from."""
    if args.base_url:
        return args.base_url, 'custom'
    source = args.source
    if source is None:
        options = [(key, f'{value["title"]}\n       {value["url"]}')
                   for key, value in MIRRORS.items()]
        source = ask_choice('Where should the datasets be downloaded from?', options,
                            default=DEFAULT_SOURCE) if interactive else DEFAULT_SOURCE
    if source not in MIRRORS:
        raise DatasetError('Unknown source "{}". Known sources: {}'.format(
            source, ', '.join(MIRRORS)))
    return MIRRORS[source]['url'], source


def build_parser():
    parser = argparse.ArgumentParser(
        prog='download_train_ds',
        description='Download the JPEG AI training and validation datasets. Run without '
                    'options to be asked what is needed.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    asked = parser.add_argument_group('answers (asked interactively when not given)')
    asked.add_argument('--source', choices=sorted(MIRRORS), default=None,
                       help='Mirror to download from')
    asked.add_argument('--datasets', choices=['train', 'validation', 'both'], default=None,
                       help='Which datasets are needed')
    asked.add_argument('--natural', choices=['full', 'patches', 'none'], default=None,
                       help='Form of the natural training content: full-size images or '
                            'cropped patches')
    asked.add_argument('--validation', choices=['cropped', 'full', 'all', 'none'],
                       default=None, help='Form of the validation set to download')
    asked.add_argument('--extras', type=parse_extras, default=None,
                       metavar='all|none|NAME,NAME',
                       help='Extra training datasets (screen content, high frequency, ...)')
    asked.add_argument('--unpack', dest='unpack', action='store_true', default=None,
                       help='Unpack the archives after downloading')
    asked.add_argument('--no-unpack', dest='unpack', action='store_false',
                       help='Only download, do not unpack')
    asked.add_argument('--remove-archives', dest='remove_archives', action='store_true',
                       default=None, help='Delete each archive once it has been unpacked')
    asked.add_argument('--keep-archives', dest='remove_archives', action='store_false',
                       help='Keep the archives after unpacking')
    asked.add_argument('--existing', choices=['resume', 'redownload', 'skip'], default=None,
                       help='What to do with archives that are already on disk: finish the '
                            'incomplete ones, fetch everything again, or leave them alone')
    asked.add_argument('--reunpack', dest='reunpack', action='store_true', default=None,
                       help='Unpack archives whose content is already unpacked')
    asked.add_argument('--no-reunpack', dest='reunpack', action='store_false',
                       help='Leave archives that are already unpacked alone')

    where = parser.add_argument_group('locations')
    where.add_argument('--data-dir', dest='data_dir', default=None,
                       help=f'Directory the datasets are unpacked into (default: '
                            f'{os.path.join(REPO_ROOT, "data")})')
    where.add_argument('--archives-dir', dest='archives_dir', default=None,
                       help='Where the downloaded archives are kept (default: <data-dir>)')
    where.add_argument('--base-url', dest='base_url', default=None,
                       help='Download from this URL instead of one of the known mirrors')
    where.add_argument('--depth', type=int, default=2,
                       help='How deep to walk the published directory index')

    run = parser.add_argument_group('running')
    run.add_argument('--answers', default=None, metavar='FILE',
                     help='Read the answers from a file written by --save-answers')
    run.add_argument('--save-answers', dest='save_answers', default=None, metavar='FILE',
                     help='Save the answers so the same run can be repeated')
    run.add_argument('-y', '--yes', action='store_true',
                     help='Do not ask for the final confirmation')
    run.add_argument('--status', action='store_true',
                     help='Report what is already on disk and exit, without connecting')
    run.add_argument('--check', action='store_true',
                     help='Compare everything the mirror offers with what is on disk, then '
                          'exit: what is complete, what is short and by how much, what is '
                          'already unpacked')
    run.add_argument('--list-remote', dest='list_remote', action='store_true',
                     help='Print the archives the mirror offers, and how each was classified')
    run.add_argument('--dry-run', dest='dry_run', action='store_true',
                     help='Stop after the summary, without downloading')
    run.add_argument('--retries', type=int, default=3,
                     help='Attempts per archive before giving up')
    run.add_argument('--strip', type=int, default=0, metavar='N',
                     help='Drop N leading path components when unpacking')
    run.add_argument('--force-lists', dest='force_lists', action='store_true',
                     help='Rewrite the file lists even when they are already there')
    run.add_argument('--no-verify-checksums', dest='verify_checksums', action='store_false',
                     help='Do not check downloads against published checksums')
    run.add_argument('--no-head', dest='use_head', action='store_false',
                     help='Trust the sizes in the index instead of asking with HEAD')
    run.add_argument('--no-space-check', dest='no_space_check', action='store_true',
                     help='Do not refuse to start when free disk space looks insufficient')
    run.add_argument('-v', '--verbose', action='store_true', help='Report each index page read')
    run.add_argument('-q', '--quiet', action='store_true', help='Less progress output')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.answers:
        load_answers(args.answers, args)
    data_dir = os.path.abspath(args.data_dir or os.path.join(REPO_ROOT, 'data'))
    archives_dir = os.path.abspath(args.archives_dir or data_dir)

    if args.status:
        return print_status(data_dir, archives_dir) or 0

    interactive = sys.stdin.isatty()
    if not interactive and not (args.list_remote or args.check) and args.datasets is None \
            and not args.yes:
        raise DatasetError(
            'There is no terminal to ask on. Pass the answers as options '
            '(--source, --datasets, --natural, --extras), use --answers FILE, or pass --yes '
            'to accept the defaults.')

    base_url, source = resolve_source(args, interactive)
    print(f'\nReading the catalogue from {base_url}')
    try:
        files, checksums = crawl(base_url, depth=args.depth, use_head=args.use_head,
                                 verbose=args.verbose)
    except DatasetError as error:
        print(f'  the directory index could not be read: {error}')
        files, checksums = list(), dict()
    if not files:
        # Some mirrors serve the files but not a listing of them.
        print(f'  falling back to the published {KNOWN_CATALOGUE_VERSION} catalogue')
        files = catalogue_from_known(base_url, use_head=args.use_head)
    archives = group_volumes(files)
    if not archives:
        raise DatasetError(f'No archives found under {base_url}. Check the address, or point '
                           '--base-url at the right directory and raise --depth.')
    print(f'Found {format_total_sets(archives)} in {len(files)} file(s)')
    receipt = load_receipt(data_dir)

    if args.list_remote:
        rows = [[x.label(), x.kind, x.group, format_size(x.size)] for x in archives]
        print()
        print(format_table(['archive', 'content', 'dataset', 'size'], rows))
        return 0

    if args.check:
        # Compare everything the mirror offers with what is already here, and stop.
        print()
        print_existing(inspect_local(archives, archives_dir, receipt))
        print(f'\nArchives in {display_path(archives_dir)}, '
              f'datasets in {display_path(data_dir)}')
        return 0

    selection = select_files(args, archives, interactive)
    if not selection:
        print('\nNothing selected, nothing to do.')
        return 0

    if args.unpack is None:
        args.unpack = ask_yes_no('\nUnpack the archives after downloading?',
                                 default=True) if interactive else True
    if args.unpack:
        require_sevenzip(selection)
    if args.remove_archives is None:
        args.remove_archives = ask_yes_no('Delete each archive once it is unpacked?',
                                          default=False) \
            if (interactive and args.unpack) else False

    # What is already downloaded or unpacked, and what to do about it.
    states = inspect_local(selection, archives_dir, receipt)
    policy = ask_existing_policy(args, states, interactive)
    reunpack = ask_reunpack(args, states, interactive)

    if not confirm_plan(args, selection, states, policy, data_dir, interactive):
        print('Cancelled.')
        return 1
    if args.save_answers:
        save_answers(args.save_answers, args, source)
    if args.dry_run:
        print('\nDry run: nothing was downloaded.')
        return 0

    os.makedirs(archives_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print('\nDownloading:')
    for state in states:
        for remote, _, _ in state.volumes:
            outcome = download_file(remote, archives_dir, retries=max(1, args.retries),
                                    quiet=args.quiet, policy=policy)
            checksum_url = checksums.get(remote.name)
            if args.verify_checksums and checksum_url and outcome in ('fresh', 'resumed'):
                if verify_checksum(os.path.join(archives_dir, remote.name), checksum_url,
                                   quiet=args.quiet) is False:
                    raise DatasetError(f'{remote.name} does not match its published '
                                       'checksum; delete it and download again.')

    if args.unpack:
        print('\nUnpacking:')
        touched, created = set(), set()
        for state in inspect_local(selection, archives_dir, receipt):
            archive = state.archive
            if not state.is_complete:
                print(f'  {archive.name}: incomplete, not unpacked')
                continue
            if state.unpacked and state.unpacked_matches and not reunpack:
                print(f'  {archive.name}: already unpacked, left as it is')
                touched.add(plan_for(archive).directory)
                continue
            plan = plan_for(archive)
            dest = os.path.join(data_dir, plan.directory)
            written, roots = extract_archive(
                os.path.join(archives_dir, archive.first_volume.name), dest, plan,
                strip=args.strip, quiet=args.quiet)
            touched.add(plan.directory)
            receipt[archive.name] = {
                'into': plan.directory,
                'files': written,
                'size': archive.size,
                'volumes': len(archive.volumes),
                'unpacked_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            save_receipt(data_dir, receipt)
            if not plan.flatten:
                created |= {(plan.directory, x) for x in roots
                            if not x.lower().endswith(IMAGE_SUFFIXES)}
            if args.remove_archives:
                for remote, path, _ in state.volumes:
                    if os.path.isfile(path):
                        os.remove(path)
        if created:
            print('\nDirectories created inside the datasets (compare them with the paths in '
                  'cfg/training_list/Q*_training_list.txt):')
            for directory, root in sorted(created):
                print(f'  {directory}/{root}/')
        made = generate_lists(data_dir, touched, force=args.force_lists)
        if made:
            print('\nFile lists:')
            for path, count in made:
                print(f'  {display_path(path)}: {count} entries')

    print()
    print_status(data_dir, archives_dir)
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
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
