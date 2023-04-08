#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script: extract statistics from folder of trisurf simulation to HDF.

Run from command line to extract data from the otherwise fairly opaque vtus.
see main

Created on Sun Aug 15 16:18:32 2021

@author: yoav
"""

import argparse
import os
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from small_functions import valid_simulation_dict, str_to_slice
from file import get_folder_params, OpenVtuReadingFrame
from numba_numeric import (calculate_statistic_new_ww,
                                 calculate_statistic_new_w,
                                 calculate_statistic_new,
                                 calculate_statistic_old_w,
                                 calculate_statistic_old)
from vtu import xml_to_np, xml_to_2_np, PyVtu


def get_statistics_from_vtu(vtu_path, v=False, w=1, entered_numba=[0]):
    """Get main statistics from a single .vtu file.

    Takes vtu_location, the path of the vtu file, and v, the verbosity flag
    extract the geometry and calculates volume, area, gyration eigenvalues,
    active bond fraction, mean curvature, mean and std cluster size, and
    cmc-bare vesicle perimeter
    Optionally, takes -w and writes a histogram file
    and -v and being verbose
    """
    vt = PyVtu(vtu_path,load_all=False)  
    if v:
        print(f"new-typed vtu={v.new}")
        if not entered_numba[0]:
            print("Entering numba-compiled function for the first time.")
            print("please wait...")
        entered_numba[0] = 1

    if v.new:
        if w == 0 or w is None:
            out = calculate_statistic_new(v.type, v.pos,
                                          v.blist, v.tlist, v.e, v.force)
            return v.new, out
        if w == 1:
            out, df = calculate_statistic_new_w(v.type, v.pos,
                                                v.blist, v.tlist, v.e, v.force)
            return v.new, out, df
        if w == 2:
            out, df1, df2 = calculate_statistic_new_ww(v.type, v.pos,
                                                       v.blist, v.tlist,
                                                       v.e, v.force)
            return v.new, out, df1, df2
        else:
            raise ValueError(f"{w=}: couldn't find statistics type (0,1,2)")
            return v.new
    else:
        if w:
            outs, df = calculate_statistic_old_w(v.c0>0, v.pos, v.blist, 
                                                 v.tlist, v.e)
            return v.new, outs, df
        else:
            outs = calculate_statistic_old(v.c0>0, v.pos, v.blist, v.tlist)
            return v.new, outs


def get_statistics_from_vtu_old(vtu_path, v=False, w=1, entered_numba=[0]):
    """Get main statistics from a single .vtu file.

    Takes vtu_location, the path of the vtu file, and v, the verbosity flag
    extract the geometry and calculates volume, area, gyration eigenvalues,
    active bond fraction, mean curvature, mean and std cluster size, and
    cmc-bare vesicle perimeter
    Optionally, takes -w and writes a histogram file
    and -v and being verbose
    """
    # Load geometry from file
    # load xml and all relevant code
    new_ver_vtu = False
    tree = ET.parse(vtu_path)
    curr_node = tree.find('.//*[@Name=\"type\"]')
    if curr_node is not None:
        # new vtus: load all new nodes
        new_ver_vtu = True
        nodetype = xml_to_np(curr_node, dtype=int)
        # curr_node = tree.find('.//*[@Name=\"curvature\"]')
        # c = xml_to_np(curr_node)
        # curr_node = tree.find('.//*[@Name=\"normal\"]')
        # v_normals = xml_to_np(curr_node)
        curr_node = tree.find('.//*[@Name=\"spontaneous_curvature\"]')
        c0 = xml_to_np(curr_node)
        if (nodetype == nodetype[0]).all():
            # attempt to reconstruct different vertices
            if (c0 != 0).any():
                nodetype[c0 != 0] |= 2 | 256  # added "active"
                v and print('added active type')
            else:
                # try w?
                curr_node = tree.find('.//*[@Name=\"bonding_strength\"]')
                w_bond = xml_to_np(curr_node)
                if (w_bond != w_bond[0]).any():
                    nodetype[w_bond != w_bond[0]] |= 1 | 256
                    v and print('added bonding type')
        curr_node = tree.find('.//*[@Name=\"force\"]')
        force = xml_to_np(curr_node)

    else:
        # old style vtus: no type, determined by spontaneous curvature
        curr_node = tree.find('.//*[@Name=\"spontaneous_curvature\"]')
        c0 = xml_to_np(curr_node)
        active = c0 > 0

    curr_node = tree.find('.//*[@Name=\"bending_energy\"]')
    bending_E = xml_to_np(curr_node)
    curr_node = tree.find('.//*[@Name=\"Koordinate tock\"]')
    pos = xml_to_np(curr_node)
    curr_node = tree.find('.//*[@Name=\"connectivity\"]')
    blist, tlist = xml_to_2_np(curr_node)

    if v:
        print(f"new-typed vtu={new_ver_vtu}")
        if not entered_numba[0]:
            print("Entering numba-compiled function for the first time.")
            print("please wait...")
        entered_numba[0] = 1

    if new_ver_vtu:
        if w == 0 or w is None:
            out = calculate_statistic_new(nodetype, pos,
                                          blist, tlist, bending_E, force)
            return new_ver_vtu, out
        if w == 1:
            out, df = calculate_statistic_new_w(nodetype, pos,
                                                blist, tlist, bending_E, force)
            return new_ver_vtu, out, df
        if w == 2:
            out, df1, df2 = calculate_statistic_new_ww(nodetype, pos,
                                                       blist, tlist,
                                                       bending_E, force)
            return new_ver_vtu, out, df1, df2
        else:
            raise ValueError(f"{w=}: couldn't find statistics type (0,1,2)")
            return new_ver_vtu
    else:
        if w:
            outs, df = calculate_statistic_old_w(active, pos, blist, tlist,
                                                 bending_E)
            return new_ver_vtu, outs, df
        else:
            outs = calculate_statistic_old(active, pos, blist, tlist)
            return new_ver_vtu, outs


def main():
    """Strat script function.

    Does everything. see argparse help
    """
    parser = argparse.ArgumentParser(
        description='''Get statistics of major order parameters from vtu
                files.
                ''')
    parser.add_argument("input", nargs='+',
                        help='main directory, or simulation subdirectories,'
                        ' or general source of vtu files')
    parser.add_argument("-m", "--mode", choices=['main_dir', 'sim_dirs',
                                                 'vtu_source'],
                        default='main_dir',
                        help='Program mode:\n'
                        'main_dir(default): input is assumed to be the main '
                        'directory, containing simulations in subdirectories: '
                        'goes to each subdirectory and sum statistics\n'
                        'sim_dir: input is assumed to be list of directories, '
                        'containing simulations: goes to each '
                        'one and sum statistics for each\n'
                        'vtu_source: input is assumed to be . vtu files and '
                        'directories containing .vtu files: extract '
                        'all vtus and sum statistics together\n'
                        )
    parser.add_argument("-s", "--slice", nargs='?',
                        help="slice of input start:stop:step\n",
                        type=str,
                        default=None, const=':')
    parser.add_argument("-o", "--output-name", help='name of store',
                        default='pystatistics.h5')
    parser.add_argument("-v", "--verbose", help='increase output verbosity',
                        action="count")
    parser.add_argument("-w", "--write-cluster",
                        help='-w: write cluster dataframe,'
                        '-ww: write typed cluster dataframes',
                        action="count")
    parser.add_argument("-t", "--trust",
                        help=('take variable from'
                              ' folder\'s parameter file'),
                        action='store_true')
    parser.add_argument("-r", "--redo_existing_store",
                        help='redo over existing store, otherwise skip',
                        action='store_true')
    args = parser.parse_args()

    out_name = args.output_name  # name of file to create
    # moved to HDF
    out_name = os.path.splitext(out_name)[0] + '.h5'
    if args.verbose is not None:
        v = args.verbose > 0  # ease verbosity checks
        vv = args.verbose > 1
    else:
        v, vv = False, False
    w = args.write_cluster
    t = args.trust
    mode = args.mode
    if args.slice is not None:
        # copied from stack overflow: split by :, place in slice constructor
        slc = str_to_slice(args.slice)
        vv and print('using', slc)
    else:
        slc = None
    redo = args.redo_existing_store

    if mode == 'main_dir':
        main_dir = args.input[0]
        directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
    elif mode == 'sim_dirs':
        directories = []
        for x in args.input:
            directories.extend([os.path.realpath(y) for y in glob(x)])

    if not t:
        if mode == 'sim_dirs':
            main_dir = os.path.dirname(directories[0])
            # check for common parent
            for directory in directories[1:]:
                if os.path.dirname(directory) != main_dir:
                    t = True  # we have to trust folders to have a valid params
    if not t:
        parameters = get_folder_params(main_dir, 'parameters')
    else:
        parameters = None
        # with open(os.path.join(main_dir, 'parameters'), 'r') as f:
        #     parameters = json.load(f)
        # total_length = reduce(lambda x, y: x*y,
        #                       (len(x) for x in parameters.values()))

    sim_directories = []
    for path in directories:
        if os.path.isdir(path):
            param_idx = valid_simulation_dict(path)
            if param_idx or mode=='sim_dirs':
                sim_directories.append((path, param_idx))
    sim_directories.sort()
    if slc is not None:
        sim_directories = sim_directories[slc]

    # statistic dataframe headers
    stat_header = ["vtu", "volume", "area", "radius",
                   "lambda1", "lambda2", "lambda3", "bond_ratio", "mean_h",
                   "perim", "mean_cluster_size", "std_cluster_size",
                   "force_per_vertex", "mean_cluster_size_per_vertex",
                   "std_cluster_size_per_vertex", ]
    df1_header = ['cluster_size', 'bending_energy',
                  'lambda1', 'lambda2', 'lambda3', 'perim',
                  "fx", "fy", "fz", 'id']
    df2_header = ['type', 'cluster_size', 'bending_energy',
                  'lambda1', 'lambda2', 'lambda3', 'perim',
                  'fx', 'fy', 'fz', 'id']

    # old statistic dataframe headers
    stat_header_old = ["vtu", "volume", "area", "radius",
                       "lambda1", "lambda2", "lambda3", "bond_ratio", "mean_h",
                       "perim", "mean_cluster_size", "std_cluster_size",
                       "mean_cluster_size_per_vertex",
                       "std_cluster_size_per_vertex", ]
    df1_header_old = ['cluster_size', 'bending_energy',
                      'lambda1', 'lambda2', 'lambda3', 'perim', 'id']

    v and print('directories: ')  # if v: print
    v and print('-', '\n- '.join(x[0]+' '+str(x[1]) for x in sim_directories))

    #############
    # main loop #
    #############

    if mode == 'main_dir' or mode == 'sim_dirs':
        for folder, param_idx in sim_directories:
            v and print('In ', folder, ':')
            v and print('-', '\n- '.join(os.listdir(folder)))

            store_path = os.path.join(folder, out_name)

            if redo and os.path.isfile(store_path):
                os.remove(store_path)
            elif os.path.isfile(store_path):
                with pd.HDFStore(store_path) as store:
                    try:
                        if store["done"]['done']:
                            continue  # skip any folder that was already done
                    except KeyError:
                        pass

            folder_params = get_folder_params(folder, 'parameters',
                                              parameters, param_idx)
            if not t:
                if parameters is not None:
                    assert({key: parameters[key][idx] for
                            key, idx in param_idx.items()} == folder_params)
            if folder_params is None:
                raise ValueError('No parameters could be found! reconsider -t')

            v and print("parameters:")
            v and print(folder_params)

            with pd.HDFStore(store_path, 'w') as store:
                store["folder_data"] = pd.Series(folder_params)
                vv and print("open store", os.path.split(store.filename)[-1])
                store["done"] = pd.Series({'done': False})

                with OpenVtuReadingFrame(folder) as f:
                    for i, file in enumerate(f):
                        filename = os.path.split(file)[-1]
                        v and print('file:', filename)

                        if vv or (v and i == 0):
                            out = get_statistics_from_vtu(file, v, w)
                        else:
                            out = get_statistics_from_vtu(file, False, w)
                        _w = w
                        if not out[0] and _w == 2:
                            # found old version: doesn't support w==2
                            _w = 1

                        if _w == 0 or _w is None:
                            is_new_ver, pystats = out
                        elif _w == 1:
                            is_new_ver, pystats, df1_data = out
                            if is_new_ver:
                                df1 = pd.DataFrame(dict(zip(df1_header,
                                                            df1_data)))
                            else:
                                df1 = pd.DataFrame(dict(zip(df1_header_old,
                                                            df1_data)))
                            store[f"df_{i:06}"] = df1
                        elif _w == 2:
                            is_new_ver, pystats, df1_data, df2_data = out
                            if not is_new_ver:
                                raise ValueError("old version has no type!")
                            df1 = pd.DataFrame(dict(zip(df1_header, df1_data)))
                            store[f"df_{i:06}"] = df1
                            df2 = pd.DataFrame(dict(zip(df2_header, df2_data)))
                            store[f"df2_{i:06}"] = df2
                            # write df

                        if is_new_ver:
                            df = pd.DataFrame(((filename, *pystats),
                                               ),
                                              # tuple-in-tuple for single row
                                              columns=stat_header, index=[i])
                        else:
                            df = pd.DataFrame(((filename, *pystats),
                                               ),
                                              # tuple-in-tuple for single row
                                              columns=stat_header_old,
                                              index=[i])
                        store.append('main_data', df)

                        if i % 50 == 0 and i != 0:
                            # save every 50 iterations
                            store.flush()
                    store["done"] = pd.Series({'done': True})

    elif mode == 'vtu_source':
        all_vtus = []
        for vtu_in in args.input:

            v and print('discerning input ', vtu_in, '\n')

            if os.path.isdir(vtu_in):
                v and print('directory: extracting files')
                with os.scandir(vtu_in) as it:
                    strings = [entry.path for entry in it]
            elif os.path.isfile(vtu_in):
                v and print('file: extracting')
                strings = [vtu_in]  # to get same handling as directory case
            else:
                raise ValueError('input ', args.input,
                                 'is not file or directory')

            # get only .vtu files
            vtus = [s for s in strings if os.path.splitext(s)[-1] == '.vtu']

            v and print("got ", len(vtus), " files", vtus[:5], "...")
            all_vtus.extend(vtus)

        # sort alphabetically:
        all_vtus.sort()
        # do the per-folder stuff
        raise NotImplementedError("vtu_source not implemented")

    v and print('done!')


if __name__ == "__main__":
    main()
