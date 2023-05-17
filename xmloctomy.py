# -*- coding: utf-8 -*-
"""Reduce file sizes of VTUs. See command line help.

Vtu-changing kernel in vtu.py (_xmloctomy) """
import argparse
import glob
import os
import vtu
import tarfile
from file import OpenVtuReadingFrame
from small_functions import valid_simulation_dict, is_nondigit, str_to_slice


def folder_xmloctomy(folder, keep_last=True, skip_tenth=True,
                     streamline_tape=True, rmfields=None, tar_name=None,
                     tar_mode='w:gz', verbose=False):
    """Remove unnessecary xml tags from most vtus in this folder.

    default: keep last vtu (so it can be continoued), skips every tenth,
    streamlines the tape
    """
    with OpenVtuReadingFrame(folder) as f:
        vtus = list(f)
        if keep_last:
            vtus.pop()
        if skip_tenth:
            vtus = [vtu for i, vtu in enumerate(vtus) if i % 10 != 0]
        if verbose:
            print(f"in {folder}, xmloctomy on {vtus}")
        for vfile in vtus:
            vtu._xmloctomy(vfile, streamline_tape, rmfields)
        if tar_name is not None:
            tar_path = os.path.join(folder, tar_name)
            with tarfile.open(tar_path, tar_mode) as tarf:
                for vfile in f:
                    tarf.add(os.path.split(vfile)[1])
        # extracted files will deleted at exit


def main():
    """Do Main function."""
    parser = argparse.ArgumentParser(
        description='''Reduce size of vtus in the folder.

            removes <trisurf> tag and streamline tape from 9/10 of the .vtus.
            reducing their size
            ''')
    parser.add_argument("input", nargs='+',
                        help='main directory, or simulation subdirectories,'
                        )
    parser.add_argument("-m", "--mode", choices=['main_dir', 'sim_dirs'],
                        default='main_dir',
                        help='Program mode:\n'
                        'main_dir(default): input is assumed to be the main '
                        'directory, containing simulations in subdirectories: '
                        'goes to each subdirectory\n'
                        'sim_dir: input is assumed to be list of directories, '
                        'containing simulations: goes to each '
                        'one\n'
                        )
    parser.add_argument("-s", "--slice", nargs='?',
                        help="slice of input start:stop:step\n",
                        type=str,
                        default=None, const=':')
    parser.add_argument("-v", "--verbose", help='increase output verbosity',
                        action="count")
    parser.add_argument("-a", "--all",
                        help=('all vtu files'),
                        action='store_false')
    parser.add_argument("-k", "--keep-tape",
                        help='don\'t streamline tape',
                        action='store_false')
    parser.add_argument("-r", "--remove-fields", nargs='?',
                        help="""remove fields from the file\n
                        'debug' will remove various debug fields (like eig0, eig1)
                        'reduce_new' will remove more fields (like gaussian curvature)
                        'all' will remove almost all new fields (like type)
                        'fld1,fld2,fld3,...' will remove the specified fields (by xml-element name!)""",
                        type=str,
                        default=None, const='')
    parser.add_argument("-t", "--tar-archive", nargs="?",
                        help="""tar all vtus in a folder to a tar archive.
                        Existing tar archive are not affected in normal operation: save to tar to record them as well.""",
                        type=str, default=None, const='timesteps.tar.gz')
    args = parser.parse_args()

    # argument processing
    if args.verbose is not None:
        v = args.verbose > 0  # ease verbosity checks
        vv = args.verbose > 1
    else:
        v, vv = False, False
    mode = args.mode
    if args.slice is not None:
        slc = str_to_slice(args.slice)
        vv and print('using', slc)
    else:
        slc = None
    rmfields = args.remove_fields
    if rmfields is not None:
        rmfields = rmfields.split(",")
        if v: print("removing fields:",*rmfields)

    # find files
    if mode == 'main_dir':
        main_dir = args.input[0]
        directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
        sim_directories = []
        for path in directories:
            if os.path.isdir(path) and not is_nondigit(path):
                param_idx = valid_simulation_dict(path)
                if param_idx:
                    sim_directories.append(path)
        directories = sim_directories
    elif mode == 'sim_dirs':
        directories = []
        for x in args.input:
            directories.extend(glob.glob(x))
        directories = [os.path.realpath(x) for x in directories]

    # get directories
    directories.sort()
    if slc is not None:
        directories = directories[slc]

    # xmloctomy on each directory
    if v: print(directories)
    for folder in directories:
        v and print(folder)
        folder_xmloctomy(os.path.realpath(folder), True, args.all,
                         args.keep_tape, rmfields, args.tar_archive,
                         verbose=v)


if __name__ == "__main__":
    main()