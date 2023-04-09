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
from vtu import PyVtu


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

def get_statistics_from_vtu(vtu_path, v=False, w=1, entered_numba=[0]):
    """Get main statistics from a single .vtu file.

    Takes vtu_location, the path of the vtu file, and v, the verbosity flag
    extract the geometry and calculates volume, area, gyration eigenvalues,
    active bond fraction, mean curvature, mean and std cluster size, and
    cmc-bare vesicle perimeter
    Optionally, takes -w and writes a histogram file
    and -v and being verbose
    """
    vtu = PyVtu(vtu_path,load_all=False)  
    if v:
        print(f"new-typed vtu={vtu.new}")
        if not entered_numba[0]:
            print("Entering numba-compiled function for the first time.")
            print("please wait...")
        entered_numba[0] = 1

    if vtu.new:
        if w == 0 or w is None:
            out = calculate_statistic_new(vtu.type, vtu.pos,
                                          vtu.blist, vtu.tlist, vtu.e, vtu.force)
            return vtu.new, out
        if w == 1:
            out, df = calculate_statistic_new_w(vtu.type, vtu.pos,
                                                vtu.blist, vtu.tlist, vtu.e, vtu.force)
            return vtu.new, out, df
        if w == 2:
            out, df1, df2 = calculate_statistic_new_ww(vtu.type, vtu.pos,
                                                       vtu.blist, vtu.tlist,
                                                       vtu.e, vtu.force)
            return vtu.new, out, df1, df2
        else:
            raise ValueError(f"{w=}: couldn't find statistics type (0,1,2)")
    else:
        if w:
            outs, df = calculate_statistic_old_w(vtu.c0>0, vtu.pos, vtu.blist, 
                                                 vtu.tlist, vtu.e)
            return vtu.new, outs, df
        else:
            outs = calculate_statistic_old(vtu.c0>0, vtu.pos, vtu.blist, vtu.tlist)
            return vtu.new, outs

def args_modify_vars(pargs):
    """Process the arguments parsed and attach them to back to the struct"""
    pargs.hdf_name = os.path.splitext(pargs.output_name)[0] + '.h5'
    if pargs.verbose is not None:
        pargs.v = pargs.verbose > 0  # ease verbosity checks
        pargs.vv = pargs.verbose > 1
    else:
        pargs.v = None
        pargs.vv = None
    pargs.w = pargs.write_cluster
    # args.t = args.trust
    # args.mode = args.mode
    if pargs.slice is not None:
        pargs.slc = str_to_slice(pargs.slice)
    else:
        pargs.slc = None
    # args.redo = args.redo_existing_store
    return pargs

def verbose_print(v,*args,**kwargs):
    if v:
        print(*args,**kwargs)



def fill_HDF_from_sims(parameters,vtus,store,pargs):
    store["folder_data"] = pd.Series(parameters)
    store["done"] = pd.Series({'done': False})

    for i, file in enumerate(vtus):
        filename = os.path.split(file)[-1]
        verbose_print(pargs.vv,'file:', filename)

        if pargs.vv or (pargs.v and i == 0):
            out = get_statistics_from_vtu(file, pargs.v, pargs.w)
        else:
            out = get_statistics_from_vtu(file, False, pargs.w)
        _w = pargs.w
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
    pargs = parser.parse_args()

    pargs = args_modify_vars(pargs)
    verbose_print(pargs.vv,'using slice', pargs.slc)

    if pargs.mode == 'main_dir':
        main_dir = pargs.input[0]
        directories = [os.path.join(main_dir, x) for x in os.listdir(main_dir)]
    elif pargs.mode == 'sim_dirs':
        directories = []
        for x in pargs.input:
            directories.extend([os.path.realpath(y) for y in glob(x)])

    if not pargs.trust:
        if pargs.mode == 'sim_dirs':
            main_dir = os.path.dirname(directories[0])
            # check for common parent
            for directory in directories[1:]:
                if os.path.dirname(directory) != main_dir:
                    pargs.trust = True  # we have to trust folders to have a valid params
    if not pargs.trust:
        parameters = get_folder_params(main_dir, 'parameters.json')
    else:
        parameters = None

    sim_directories = []
    for path in directories:
        if os.path.isdir(path):
            param_idx = valid_simulation_dict(path)
            if param_idx or pargs.mode=='sim_dirs':
                sim_directories.append((path, param_idx))
    sim_directories.sort()
    if pargs.slc is not None:
        sim_directories = sim_directories[pargs.slc]

    verbose_print(pargs.v,'directories: ')
    verbose_print(pargs.v,'-', '\n- '.join(x[0]+' '+str(x[1]) for x in sim_directories))

    #############
    # main loop #
    #############

    if pargs.mode == 'main_dir' or pargs.mode == 'sim_dirs':
        for folder, param_idx in sim_directories:
            verbose_print(pargs.v,'In ', folder, ':')
            store_path = os.path.join(folder, pargs.hdf_name)

            if pargs.redo_existing_store and os.path.isfile(store_path):
                os.remove(store_path)
            elif os.path.isfile(store_path):
                with pd.HDFStore(store_path) as store:
                    try:
                        if store["done"]['done']:
                            continue  # skip any folder that was already done
                    except KeyError:
                        pass

            folder_params = get_folder_params(folder, 'parameters.json',
                                              parameters, param_idx)
            if not pargs.trust:
                if parameters is not None:
                    assert({key: parameters[key][idx] for
                            key, idx in param_idx.items()} == folder_params)
            if folder_params is None:
                raise ValueError('No parameters could be found! reconsider -t option')

            verbose_print(pargs.v,"parameters:", parameters)

            with pd.HDFStore(store_path, 'w') as store:
                verbose_print(pargs.vv,"open store", os.path.split(store.filename)[-1])
                with OpenVtuReadingFrame(folder) as f:
                    vtus = list(f)
                    verbose_print(pargs.v,'-', '\n- '.join([os.path.split(file)[-1] for file in vtus]))
                    fill_HDF_from_sims(parameters,vtus,store,pargs)
                    

    elif pargs.mode == 'vtu_source':
        all_vtus = []
        for vtu_in in pargs.input:
            verbose_print(pargs.v,'discerning input ', vtu_in, '\n')
            if os.path.isdir(vtu_in):
                verbose_print(pargs.v,'directory: extracting files')
                with os.scandir(vtu_in) as it:
                    strings = [entry.path for entry in it]
            elif os.path.isfile(vtu_in):
                verbose_print(pargs.v,'file: extracting')
                strings = [vtu_in]  # to get same handling as directory case
            else:
                raise ValueError('input ', pargs.input,
                                    'is not file or directory')

            # get only .vtu files
            vtus = [s for s in strings if os.path.splitext(s)[-1] == '.vtu']

            verbose_print(pargs.v,"got ", len(vtus), " files", vtus[:5], "...")
            all_vtus.extend(vtus)

        # sort alphabetically:
        all_vtus.sort()
        # do the per-folder stuff
        with pd.HDFStore(store_path, 'w') as store:
            fill_HDF_from_sims({},vtus,store,pargs)

    verbose_print(pargs.v,'done!')


if __name__ == "__main__":
    main()
