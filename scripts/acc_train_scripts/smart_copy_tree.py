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

import hashlib
import os
import stat
from shutil import Error, copy, copy2, copyfile, copystat

# copy of copytree of python 3.10, since not availabe on roma ( need dirs_exist arg )
# extended which hashlib to only copy files when hash is different


def get_hash(file):
    # hashing files in python, see https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    # BUF_SIZE is totally arbitrary, change for your app!
    BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

    md5 = hashlib.md5()

    with open(file, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def _smart_copytree(entries,
                    src,
                    dst,
                    symlinks,
                    ignore,
                    copy_function,
                    ignore_dangling_symlinks,
                    dirs_exist_ok=False):
    if ignore is not None:
        ignored_names = ignore(os.fspath(src), [x.name for x in entries])
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=dirs_exist_ok)
    errors = []
    use_srcentry = copy_function is copy2 or copy_function is copy

    for srcentry in entries:
        if srcentry.name in ignored_names:
            continue
        srcname = os.path.join(src, srcentry.name)
        dstname = os.path.join(dst, srcentry.name)
        srcobj = srcentry if use_srcentry else srcname
        try:
            # is_symlink = srcentry.is_symlink() # TODO: can it be fixed for cluster
            is_symlink = False
            if is_symlink and os.name == 'nt':
                # Special check for directory junctions, which appear as
                # symlinks but we want to recurse.
                lstat = srcentry.stat(follow_symlinks=False)
                if lstat.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT:
                    is_symlink = False
            if is_symlink:
                linkto = os.readlink(srcname)
                if symlinks:
                    # We can't just leave it to `copy_function` because legacy
                    # code with a custom `copy_function` may rely on copytree
                    # doing the right thing.
                    os.symlink(linkto, dstname)
                    copystat(srcobj, dstname, follow_symlinks=not symlinks)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occur. copy2 will raise an error
                    if srcentry.is_dir():
                        smart_copy_tree(srcobj,
                                        dstname,
                                        symlinks,
                                        ignore,
                                        copy_function,
                                        dirs_exist_ok=dirs_exist_ok)
                    else:
                        copy_function(srcobj, dstname)
            elif srcentry.is_dir():
                smart_copy_tree(srcobj,
                                dstname,
                                symlinks,
                                ignore,
                                copy_function,
                                dirs_exist_ok=dirs_exist_ok)
            else:
                if os.path.isfile(dstname):
                    if get_hash(srcobj) == get_hash(dstname):
                        continue  # ignore, if identical files already exists in destination
                # Will raise a SpecialFileError for unsupported file types

                # srcobj = srcobj.path
                # print(f"srcobj {srcobj}, {type(srcobj)}")
                # print(f"srcobj {dstname}, {type(dstname)}")
                copy_function(srcobj, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    # TODO: can it be fixed for cluster
    # try:
    #     copystat(src, dst)
    # except OSError as why:
    #     # Copying file access times may fail on Windows
    #     if getattr(why, 'winerror', None) is None:
    #         errors.append((src, dst, str(why)))
    # if errors:
    #     raise Error(errors)
    return dst


def smart_copy_tree(
        src,
        dst,
        symlinks=False,
        ignore=None,
        copy_function=copyfile,  # TODO: can it be fixed for cluster (default is copy2)
        ignore_dangling_symlinks=False,
        dirs_exist_ok=False):
    """Like shutil.copytree, but checking with hasfunction whether file in destination
    should be updated.
    Recursively copy a directory tree and return the destination directory.
    If exception(s) occur, an Error is raised with a list of reasons.
    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.
    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.
    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():
        callable(src, names) -> ignored_names
    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.
    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.
    If dirs_exist_ok is false (the default) and `dst` already exists, a
    `FileExistsError` is raised. If `dirs_exist_ok` is true, the copying
    operation will continue if it encounters existing directories, and files
    within the `dst` tree will be overwritten by corresponding files from the
    `src` tree.
    """
    # sys.audit("shutil.copytree", src, dst)
    with os.scandir(src) as itr:
        entries = list(itr)
    return _smart_copytree(entries=entries,
                           src=src,
                           dst=dst,
                           symlinks=symlinks,
                           ignore=ignore,
                           copy_function=copy_function,
                           ignore_dangling_symlinks=ignore_dangling_symlinks,
                           dirs_exist_ok=dirs_exist_ok)
