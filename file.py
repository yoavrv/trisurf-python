#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for file manipulation, file data extraction functions and classes.

CLASSES:
    open_vtu_reading_frame: context manager for extracting and iterating
                            over all vtus in a folder
    MultiOVRF: Wraps several open_vtu_reading_frames
    SimInfo: NamedTuple, represent simulation (path, param_idx, store)
    FolderInfo: Represents simulations folder (parameters)
    SimsData: Represents simulations folder 

FUNCTIONS:
    extract sim params: try hard to extract data from parameter file
    get_folder_params: try hard to extract params of directory
    get_name: generates the leading x#### name from SimInfo path

    
Created on Thu Aug 12 14:27:13 2021

@author: yoav
"""

import os
import tarfile
import json
import typing 
from glob import glob

import numpy as np
import pandas as pd

from small_functions import valid_simulation_dict


def is_tarfile_and_not_directory(x):
    """Wrap tarfile.is_tarfile but without breaking error."""
    name = os.path.split(x)
    if name.endswith('.tar') or name.endswith('.tar.gz'):
        try:
            return tarfile.is_tarfile(x)
        except IsADirectoryError:
            return False
    return False


def get_folder_params(folder, param_file_name='parameters.json',
                      parameters=None, param_idx=None) -> dict:
    """Attempt to get folder parameters dictionary from folder.

    For a simulation folder x0000_a1b2c3, load x0000/parameters.json file.
    in case of failure, derive from all simulations parameters and param_idx.
    If param_idx is None, derive from the folder name using valid_simulation_dict.

    For a simulations folder /folder/[x0000...x9999], load the folder/parameters.json
    In case of failure, derive from parameters and param_idx parameters.
    """
    folder_params = None

    try:
        with open(os.path.join(folder,param_file_name), 'r') as f:
            # json
            folder_params = json.load(f)
    except (FileNotFoundError, ValueError):
        if parameters is not None:
            if param_idx is None:
                param_idx = valid_simulation_dict(folder)
            folder_params = {key: parameters[key][idx] for
                                key, idx in param_idx.items()}
    return folder_params


class OpenVtuReadingFrame():
    """Context manager for extracting all vtus in folder.

    extract all vtus in valid tars on entry, delete them on exist
    vtus are available by iteration

    e.g.:
    with open_vtu_reading_frame('folder') as orf:
        for vtu in orf:
            # operate on each vtu
    """

    def __init__(self, folder, validate_vtu_func=None,
                 validate_tar_func=None, validate_vtu_ext=True):
        """Initialize Open reading frame for vtus.

        validating functions should act on the name,
        validate_vtu_func = lambda x: 'timestep' in x 
        will take "timestep_000000.vtu" but not "base.vtu" 
        and not "timesteps.tar.gz" (if validate_vtu_ext is True)
        validate_tar_func = lambda x: 'Xs' in x and 'old' not in x
        will take "Xs.tar.gz" but not "old_Xs.tar.gz" or "steps.tar" 
        tar is validate by tarfile.is_tarfile() so no validate_tar_ext required
        """
        self.folder = folder
        self.is_open = None
        self.is_valid_tar = validate_tar_func if validate_tar_func else lambda x: True
        is_valid_vtu = validate_vtu_func if validate_vtu_func else lambda x: True
        if validate_vtu_ext:
            self.is_valid_vtu = lambda x: (os.path.splitext(x)[-1] == '.vtu'
                                           and is_valid_vtu(x))


    def __enter__(self):
        """Entry function: Extract vtu files and remember them for deletion on exit."""
        self.is_open = True
        self.original_files = []
        self.tar_archives = []
        self.extracted_files = []
        self.all_vtus = []
        rawlist = os.listdir(self.folder)
        pathlist = [os.path.join(self.folder, x) for x in rawlist]
        self.original_files = [x for x in pathlist
                               if self.is_valid_vtu(x)]
        self.tar_archives = [x for x in pathlist
                             if is_tarfile_and_not_directory(x)
                             and self.is_valid_tar(x)]
        # untar files
        for archive in self.tar_archives:
            with tarfile.open(archive,'r') as arc:
                vtu_members = [x for x in arc.getmembers()
                                if self.is_valid_vtu(x.name)
                                and x.name
                                not in rawlist
                                and os.path.join(self.folder, x.name)
                                not in self.extracted_files
                                ]
                self.extracted_files.extend(os.path.join(self.folder, x.name)
                                            for x in vtu_members)
                arc.extractall(path=self.folder, members=vtu_members)

        # get all vtus we extracted
        self.all_vtus.extend(self.original_files)
        self.all_vtus.extend(self.extracted_files)
        self.all_vtus.sort()
        # canonicalize files to delete in case of cd in the middle:
        self.extracted_files = [os.path.realpath(x)
                                for x in self.extracted_files]
        return self

    def __iter__(self):
        """Iterate over vtu files."""
        if self.is_open:
            return iter(self.all_vtus)
        else:
            raise ValueError(f"Reading frame {self} is not open")

    def __exit__(self, *exc):
        """Exit function: delete extracted files."""
        for file in self.extracted_files:
            os.remove(file)
        self.is_open = False

    def __del__(self):
        """Delete extracted files before object is destroyed (hopefully)."""
        if self.is_open:
            self.__exit__()


class MultiOVRF():
    """Context manager for multiple vtu reading frame.

    Takes multiple simulation directories and opens a vtu reading frame
    for each, closing afterwards.
    Usefull for temporarily de-tarring a few simulations for
    paraview visualization
    """

    def __init__(self, sim_dirs, validate_vtu_func=None,
                 validate_tar_func=None, validate_vtu=True):
        self.sim_dirs = sim_dirs
        self.sim_reading_frames = []
        for _dir in sim_dirs:
            (self.sim_reading_frames.append(
                OpenVtuReadingFrame(_dir,
                                    validate_vtu_func=validate_vtu_func,
                                    validate_tar_func=validate_tar_func,
                                    validate_vtu=validate_vtu))
             )

    def __enter__(self):
        """Open each reading frame.

        To iterate: vtus are available through
        self.sim_reading_frames[i].all_vtus
        self.sim_reading_frames[i].__iter__() == all_vtus.iter()
        """
        self.iters_vtu = []
        for rf in self.sim_reading_frames:
            rf.__enter__()
        return self


    def __exit__(self, *exc):
        """Close each reading frame."""
        for rf in self.sim_reading_frames:
            rf.__exit__()


def get_name(sim_path):
    """Generate name (x####)."""
    return os.path.split(sim_path)[1][:5]


def print_params(d: dict, rmkeys=None):
    """Get string representation of dict (fitting filenames)."""
    if rmkeys is None:
        return ", ".join(f"{key}: {value:.03}" if type(value) is float
                         else f'{key}: {value}' for key, value in d.items())
    else:
        return ", ".join(f"{key}: {value:.03}" if type(value) is float
                         else f'{key}: {value}' for key, value in d.items()
                         if key not in rmkeys)

def get_timesteps(path):
    """Use glob to get all timesteps in a folder"""
    return sorted(glob(os.path.join(path, "timestep_[0-9]*.vtu")))

class SimFrame():
    """Class for representing a simulations folder for manual and programatic exploration.
    
    For a simulation folder
    /folder
    /folder/parameters.json
    /folder/x0000_a0b0c0
    /folder/x0000_a0b0c0/parameters.json
    /folder/x0000_a0b0c0/pystatistics.h5
    /folder/x0000_a0b0c0/timestep_000000.vtu
    ...
    /folder/x0000_a0b0c0/timestep_000099.vtu
    /folder/x0001_a0b0c1
    ...
    /folder/x0049_a2b4c4/timestep_000089.vtu

    SimFrame organize them in a dataframe self.df where eeach row represent a simulation
    >>> frame=SimFolder('folder')
    >>> frame.df
        a     b    c    path            timesteps   store
    0   20   2.0  -0.1  /folder/x0000   99          true
    1   20   2.0   0.0  /folder/x0001   99          true
    ...
    49  28   8.0   0.3  /folder/x0049   89          false
    
    This frame can than be queries into a subspace (frame.subspace)
    >>> frame.query_subspace('a==20 and c>0')
        a     b    c    path            timesteps   store
    1   20   2.0   0.0  /folder/x0001   99          true
    ...
    24  20   8.0   0.3  /folder/x0029   99          true

    This makes it easy to handle manually, for example
    >>> 
    """
  
    def __init__(self, main_folder, subfolder_glob='x*', timesteps_glob="timestep_[0-9]*.vtu",
                 param_file_name= 'parameters.json',store_name='pystatistics.h5'):
        """Initialize frame to represent main folder.
        
        Assumes simulations exists in subfolders x####
        from glob(./x*) (change with subfolder_glob) with timesteps x####.../timestep_######.vtu
        from glob(./timestep_[0-9]*.vtu) (change with timesteps_glob)
    
        The simulations are represented by rows in a dataframe, with the columns for each of
        the parameters of the simulation, taken from ./"parameters.json" in the folder
        (change with param_file_name), and columns for path, timeteps, and stroe for
        the path of the folder, number of timesteps, and if an hdf5 store pystatistics.h5
        exists in folder (change with store_name)

        For example
        >>> frame=SimFolder('folder')
        >>> frame.df
            a     b    c    path            timesteps   store
        0   20   2.0  -0.1  /folder/x0000   99          true
        1   20   2.0   0.0  /folder/x0001   99          true
        ...
        49  28   8.0   0.3  /folder/x0049   89          false
        """
        self.main_folder = os.path.realpath(main_folder)
        self.timesteps_glob = timesteps_glob
        self.get_timesteps = lambda path: sorted(glob(os.path.join(path, timesteps_glob)))
        self.parameters = get_folder_params(main_folder, param_file_name)
        self.parameters = self.parameters if self.parameters else {}
        self.store_name = store_name
        columns = ['path','timesteps','store', *self.parameters.keys()] # rearange them back later
        xs = sorted(glob(os.path.join(main_folder,subfolder_glob)))
        init_values = [self.main_folder,
                       np.ones(len(xs),dtype=int),
                       np.ones(len(xs),dtype=bool),
                       *(x[-1] for x in self.parameters.values()),]
        self.df = pd.DataFrame({c: v for c,v in zip(columns, init_values)})
        i = 0
        for path in xs:
            try:
                params = get_folder_params(path)
                if params is None:
                    # no parameters.json found in folder
                    params=valid_simulation_dict(path)
                    if self.parameters and params:
                        params = {key: values[params[key]] for key, values in self.parameters.items()}
                    if not params:
                        continue
                num_timesteps=len(self.get_timesteps(path))
                store = os.path.join(path, store_name)
                has_store = os.path.isfile(store)
                if params or num_timesteps or has_store:
                    self.df.loc[i,'path'] = path
                    self.df.loc[i,params.keys()]=params.values()
                    self.df.loc[i,'timesteps'] = num_timesteps
                    self.df.loc[i,'store']=has_store
                    i+=1
            except (FileNotFoundError, NotADirectoryError):
                continue
        self.df.drop(self.df.tail(len(xs)-i).index,
                     inplace = True)
        reorder_keys = [col for col in self.df.columns if col not in {'path','timesteps','store'}]
        self.df = self.df[[*reorder_keys,'path','timesteps','store']]
        self.subspace = self.df[:]

    def query_subspace(self, query_str=''):
        """Fix a subspace view of the dataframe based on a query.
        
        self.query_subspace(q) is equivalent to self.subspace=self.df.query(q)
        Useful for manual filtering and fixing parameters
        frame[i] returns rows of the subspace.
        See for_paraview for several convenient use case

        >>> all(frame.df == frame.subspace) # initially: full view
        True
        >>> frame[0]
        k 10
        f -9
        ...
        Name: 0, dtype: object
        >>> frame.query_subspace("k>20")
        >>> frame[0]
        k 20
        f -9
        ...
        Name: 4, dtype: object
        """
        if query_str is None or query_str=='':
            self.subspace = self.df[:] # full view
            return self.subspace
        self.subspace = self.df.query(query_str)
        return self.subspace

    def __getitem__(self, slc):
        """Equivalent to self.subpace.iloc[]"""
        return self.subspace.iloc[slc]
    
    def __len__(self):
        return self.subspace.__len__()
    
    def last_timesteps(self):
        return [x.path+f'timestep_{x.timesteps-1:06}.vtu' for x in self]
