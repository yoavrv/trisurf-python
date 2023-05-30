#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for chemfarm deployment scripts. please short to chest.

Created on Thu Jun 16 17:42:40 2022

package for all the functions in the scriptmaking jupyter notebooks for making
chemfarm-grade simulation folders


@author: yoav
"""
import os
import json
import datetime
from collections import namedtuple
from itertools import cycle, product, repeat
# import numpy as np


Param = namedtuple("Param", ["value", "id", "name"])


def get_pval(d, key, default_val=None):
    """Get parameter value from dictionary of Param."""
    return d.get(key, Param(default_val, None, None)).value


def iterate_params(all_param_dict):
    """Create an iterator for all parameter combinations in all_param_dict."""
    for x in product(
        *(enumerate(zip(p.value, repeat(p.name), repeat(key)))
          for key, p in all_param_dict.items())):
        yield {z[1][2]: Param(z[1][0], z[0], z[1][1]) for z in x}


def write_parameters_file(folder, param_dict):
    """Write parameters.json, recording x_i02 -> {i: 1.2, 1.3, 1.4, 1.5}."""
    param_val_dict = {key: param.value for key, param in param_dict.items()}
    with open(os.path.join(folder, "parameters.json"), "w") as f:
        json.dump(param_val_dict, f)


def make_dirname(num_sim, param_dict, numsize=(4, 2)):
    """Return name string for a directory in the style x0032_a01b02."""
    prefix = f"x{num_sim:0{numsize[0]}}_"
    postfix = "".join(f"{key}{param.id:0{numsize[1]}}"
                      for key, param in param_dict.items())
    return prefix + postfix


def modify_tapetext(text, param_dict, base_temperature=20):
    """Return modified tape text, updated with the parameters in param_dict.

    Special behavior for temperature: rescale over base_temperature.
    """
    lines = text.splitlines()
    newlines = []
    reprules = {param.name: param.value
                for param in param_dict.values() if param.name is not None}
    do_temperature = 'temperature' in reprules
    if do_temperature:
        temperature_reprules = {'w', 'f', 'F', 'xk0', 'kx2', 'pressure',
                                'xkA0', 'adhesion_strength'}
        factor = reprules['temperature']/base_temperature
    for i, line in enumerate(lines):
        name, *rest = line.split("=")
        if name in reprules:
            rest = str(reprules[name]),
        if do_temperature:
            if name in temperature_reprules:
                curr_val = float(*rest)
                rest = str(curr_val*factor),
        newlines.append("=".join((name, *rest)))
    return "\n".join(newlines)


def write_derivative_tape(folder, base_tape, param_dict):
    """Derive tape from base tape and parameter dict and write in subfolder."""
    if base_tape is not None:
        with open(base_tape, "r") as tape_file:
            text = tape_file.read()
        text = modify_tapetext(text, param_dict)
        with open(os.path.join(folder, "tape"), "w") as tape_file:
            tape_file.write(text)
        return text


def job_script_head(name, queue="long", select="1",
                    ncpus="1", mem="10000", walltime=None):
    """Head of a job script file."""
    string_1 = f"""#!/bin/bash
#
#PBS -N {name}
#PBS -j oe
#PBS -q {queue}
#PBS -m eb
#PBS -M yoav.ravid@weizmann.ac.il
#PBS -l select={select}:ncpus={ncpus}:mem={mem}mb
"""
    string_mid = f"\n#PBS -l walltime={walltime}:00:00"
    string_2 = """

# Print time and date, beginning of the simulation
date
echo `hostname`

"""
    if walltime:
        head_string = "".join((string_1, string_mid, string_2))
    else:
        head_string = "".join((string_1, string_2))
    return head_string


def job_script_main_dir():
    """Move PBS script to work in the current directory."""
    s = """### work in the PBS_O_WORKDIR ###

cd $PBS_O_WORKDIR

# now move to each dir, run trisurf
# a directory with the appropriate tape and vtu should be prepared ahead of time
"""
    return s


def job_script_trisurf(dir_name, trisurf_name,
                       start_params, continue_params):
    """Job execute trisurf for folder."""
    s = f"""cd {dir_name}
sleep 2
if [[ -e dump.bin ]] ; then
    time {trisurf_name} {continue_params} &
else
    time {trisurf_name} {start_params} &
fi
cd ..

"""
    return s


def job_script_tail():
    """Tail of a chemfarm jobL wait for all processes to finish and time."""
    return"""

# wait for all jobs to finish
wait
# print the time and date at the end
date
"""


def make_scripts(script_name, job_name, param_iterator,
                 base_tape, base_vtu=None, chunks=None,
                 mem_per_sim=1200, max_time=360, queue="idle", opmode=None,
                 trisurf_path="/home/yoavra/apps/bin/modeled_trisurf"):
    """Generate scripts."""
    scripts_out = []
    today = datetime.date.today()
    if opmode is None:
        opmode = "--force" if base_vtu is None else "--restore timestep_000000.vtu"
    jobs_i = 0
    if chunks is None:
        chunks = (24, 12)
    chunk_sizes = cycle(chunks)

    gen_note_str = f"# GENERATED BY PYTHON SCRIPT {script_name} on {today} #"
    padd_str = "#"*len(gen_note_str)
    generated_note = f"""{padd_str}
{gen_note_str}
{padd_str}

"""

    chunk_size = next(chunk_sizes)
    script_middles = []  # Text As List
    params = [*param_iterator]
    xsize = len(str(len(params)))
    for i, param_dict in enumerate(params):
        script_middles.append(job_script_trisurf(make_dirname(i, param_dict),
                                                 trisurf_path, f'{opmode} $@',
                                                 '$@'))
        if len(script_middles) == chunk_size:
            # create head, assemble into file
            script_heads = [job_script_head(f"{job_name}_{jobs_i:0{xsize}}",
                                            queue, 1, chunk_size,
                                            chunk_size * mem_per_sim,
                                            max_time),
                            generated_note,
                            job_script_main_dir()]
            script_end = [job_script_tail()]
            scripts_out.append([f"job_script_{jobs_i:0{xsize}}",
                               "".join((*script_heads, *script_middles,
                                        *script_end))])

            jobs_i += 1
            script_middles = []
            chunk_size = next(chunk_sizes)

    if script_middles:
        # write any remaining files
        # swap head
        chunk_size = len(script_middles)
        script_heads = [job_script_head(f"{job_name}_{jobs_i:0{xsize}}",
                                        queue, 1, chunk_size,
                                        chunk_size*mem_per_sim, max_time),
                        generated_note,
                        job_script_main_dir()]
        script_end = [job_script_tail()]
        scripts_out.append([f"job_script_{jobs_i:0{xsize}}",
                            "".join((*script_heads, *script_middles,
                                     *script_end))])
    return scripts_out

