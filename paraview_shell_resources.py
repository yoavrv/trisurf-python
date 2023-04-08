"""
Source to copy paste-things in the paraview python console

@author: yoav
"""

### main setup ###


if __name__=='__main__':
    print("this is not a script, it is a repository for copy-pasting into the paraview shell.")

SOURCE_PATH='SOURCE_PATH'
if path=='SOURCE_PATH':
    raise ImportError(
        'This is a text file for copy paste into the paraview python console. '
        'Please change the SOURCE_PATH variable for the location of the packages '
        'so they could be imported')
SOURCE_PATH = r'\\wsl$\Ubuntu-20.04\opt\workspace\msc_project\lab_note_and_py\trisurf-python'

if __name__=='__main__':
    raise ImportError("Not A Script!")

# dependencies
from glob import glob
import os
import numpy as np

import sys
sys.path.append(SOURCE_PATH)

import for_paraview as fp
import file
from vtu import PyVtu
from small_functions import Slice
SLC = Slice

loc_sims = r'\\cf-gpfs-fsrv\cf_work\sim\bicurvatures\23_vicsek_combined'

## wait a few minutes to load if remote
frame = file.SimFrame(loc_sims)
keys = [k for k in frame.df.keys() if k not in ['path','timesteps','store']]
if keys!=list(frame.parameters):
    print("parameters and frame disagree")

frame.query_subspace('limit your frame e.g. w==0 and k>20 and b==1')


# figure out N and M
unique_values = frame.subspace[keys].apply(np.unique,axis=0)
(keyN, N),(keyM, M) = [(k, len(l)) for k,l in unique_values.items() if len(l)!=1]
fast='x'
print(f"{N}x{M} {keyN}x{keyM}")
# (N, M, keyN, keyM), fast = (M, N, keyM, keyN), 'y'
grid = fp.GridParaview(N, M) # grid = fp.GridParaview(M, N) 
grid.link_views()

# check direction with SetActiveView(grid.views[0,3]) to see where it falls

names = [f'x{x:04}' for x in frame.subspace.index]
params = frame.subspace[keys].to_dict(orient="records")
param_texts = [file.print_params(param) for param in params]
print(param_texts)

views = list(grid.iter_views(fast=fast))  # usually, if inner runs before outer ie 1_f1w0 instead of 1_f0w1
# views = list(grid.iter_views(x=-1, y=-1, fast='y'))


## write text test!###

param_texts_2 = param_texts
# param_texts_2 = [file.print_params({keyM: param[keyM],keyN:param[keyN]}) for param in params]
tester = fp.text_test(views, param_texts_2)

# is it okay?
tester.delete()

for view in views:
    view.OrientationAxesVisibility=0
    Render(view)

slc_losho = SLC[:]
# slc_losho = SLC[:]
loSho_i = fp.make_loSho_i(views, frame, slc_losho, names, param_texts)

for i,_ in enumerate(views):  # views[1:],1):
    loSho_i(i)

grid.snap_animation()

for view in views:
    view.OrientationAxesVisibility = 0

for view, source in grid.sources.items():
    GetDisplayProperties(source, view=view).Opacity=0.7

# view specific simulation from frame subspace
fp.view_simframe_slice(frame, 0, view=None, slc=None, text=None)