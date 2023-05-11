#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module to import to paraview.

@author: yoav
"""

import os
from glob import glob

import numpy as np
import paraview.simple as ps

def snap_shot(name, res=(1920, 1080), transparent=False, base=None):
    """Take a snapshot of the current window.

    Return kwargs to retake the snapshot
    >>> kw=fp.snap_shot('over_here')
    >>> fp.snap_shot(**kw)  # retake
    """
    if base is not None:
        name = os.path.join(base, name)
    name = os.path.splitext(name)[0] + ".png"
    ps.SaveScreenshot(filename=name,
                   ImageResolution=res,
                   TransparentBackground=transparent)
    return dict(name=name, res=res, transparent=transparent)


def get_current_file():
    """Get the file associated with the current active source"""
    t = int(ps.GetAnimationScene().TimeKeeper.Time)
    files = ps.GetActiveSource().FileName
    return files[min(t,len(files)-1)]


def animation_iter(*args):
    '''Iterate over animation frames. iterate range(*args) or args[0]'''
    if type(args[0]) is int:
        it = range(*args)
    else:
        it = args[0]
    scene = ps.GetAnimationScene()
    for i in it:
        scene.AnimationTime=i
        yield i


class GridParaview():
    """Make and hold a grid view in paraview."""

    current_grid = []

    def __init__(self, N, M):
        """Initialize an NxM matrix of views."""
        self.layout = ps.GetLayout()
        self.views = np.zeros((N, M), dtype=object)
        self.sources = {}
        self.texts = {}
        self.current_grid.append(self)
        self.views[0, 0] = ps.GetViews()[-1]
        i = M
        while i > 1:
            hint = self.layout.SplitViewHorizontal(self.views[0, 0], (i-1)/i)
            self.views[0, i-1] = ps.CreateRenderView()
            ps.AssignViewToLayout(view=self.views[0 ,i-1],
                                  layout=self.layout, hint=hint)
            i -= 1

        for i in range(M):
            j = N
            while j > 1:
                hint = self.layout.SplitViewVertical(self.views[0, i], (j-1)/j)
                self.views[j-1, i] = ps.CreateRenderView()
                ps.AssignViewToLayout(view=self.views[j-1, i],
                                      layout=self.layout, hint=hint)
                j -= 1

    def link_views(self):
        """Link all views."""
        prime = self.views[0, 0]
        for i, view in enumerate(self.views.flatten()[1:]):
            name = f"Camera_Link_{i}"
            ps.AddCameraLink(view, prime, name)
            print(name)

    def snap_animation(self):
        """Snap animations to the timesteps."""
        # get animation scene
        ps.GetAnimationScene().PlayMode = 'Snap To TimeSteps'

    def change_fontsize(self, fontsize=18):
        for view in self.views.flat:
            dp = ps.GetDisplayProperties(self.texts[view], view)
            dp.FontSize = fontsize

    def disable_text_interactivity(self):
        for view in self.views.flat:
            dp = ps.GetDisplayProperties(self.texts[view], view)
            dp.Interactivity = 0

    def change_text_properties(self, **kwargs):
        for view in self.views.flat:
            dp = ps.GetDisplayProperties(self.texts[view], view)
            for key, value in kwargs.items():
                dp.SetPropertyWithName(key, value)

    def change_vesicles_properties(self, **kwargs):
        for view in self.views.flat:
            dp = ps.GetDisplayProperties(self.sources[view], view)
            for key, value in kwargs.items():
                dp.SetPropertyWithName(key, value)
    
    def color_vesicles_by(self,field):
        for view, source in self.items():
            ps.ColorBy(ps.GetDisplayProperties(source,view),
                       ('POINTS',field))
        
    def iter_views(self, x=1, y=1, fast='x'):
        """Iterate views [x0y0, x1y0, x2y0... x0y1...] (s shape).

        The native way is matrix self.view[i,j]->ith row, jth column.
        set x=-1 to reverse the direction of x [x2y0, x1y0...]
        (z shape from bottom right)
        set y=-1 to reverse the direction of y [x0y2, x1y2... x0y0...]
        (z shape from top left)
        set fast='y' to transpose the fast axis [x0y0, x0y1, x0y2...]
        (N shape from bottom left)
        i.e. starting from (x=,y=), first follow the line of fast axis
        (x=1,y=-1)->fast=x      fast=x<-(x=-1,y=-1)
          |                                    |
          v                                    v
        fast=y                              fast=y

        fast=y                              fast=y
          ^                                    ^
          |                                    |
        (x=1,y=-1)->fast=x      fast=x<-(x=-1,y=-1)
        """
        views = self.views
        views = views[::-y, ::x]
        if fast == 'y':
            views = views.T
        return views.flat


def loadShow(view, files, name, text=None, field='spontaneous_curvature',
             grid=None, **textkw):
    """Load a vtu source and display in view."""
    ps.SetActiveView(view)
    source = ps.XMLUnstructuredGridReader(FileName=files)
    ps.RenameSource(name)
    ps.ColorBy(ps.GetDisplayProperties(ps.GetActiveSource()),
            ('POINTS', field))
    if text is not None:
        textsource = ps.Text(Text=text)
    else:
        textsource = None
    if grid is not None:
        grid.sources[view] = source
        grid.texts[view] = textsource
    elif GridParaview.current_grid:
        _grid = GridParaview.current_grid[-1]
        _grid.sources[view] = source
        _grid.texts[view] = textsource
    tProp = ps.GetDisplayProperties(textsource, view)
    tProp.SetPropertyWithName("Interactivity", 0)
    if textkw:
        for key, value in textkw.items():
            tProp.SetPropertyWithName(key, value)
    ps.Show()  # needed for text and map


class Text_Views_Tester:
    """use self.delete to remove all texts."""
    def __init__(self):
        self.texts=[]

    def delete(self):
        for i in range(len(self.texts)):
            t = self.texts.pop()
            ps.Delete(t)


def text_test(views, texts):
    """Test views and texts arrays by giveing each view its text.
    
    Returns a tester object with a delete method to remove all the texts"""
    texts_tester = Text_Views_Tester()
    for i, (view, text) in enumerate(zip(views, texts)):
        ps.SetActiveView(view)
        texts_tester.texts.append(ps.Text(Text=f'{i} {text}'))
        ps.Show()
    return texts_tester


def make_load_show_i(views, sims, slc, names, texts,field="spontaneous_curvature"):
    if type(field)==str:
        getfield=lambda i: field
    else:
        getfield=lambda i: field[i]
    if len(views)==1:
        getview = lambda i: views
    else:
        getview = lambda i: views[i]
    if type(sims) is list:  # assume list of paths
        def get_timesteps(i):
            return sorted(glob(os.path.join(sims[i], "timestep_[0-9]*.vtu")))[slc]
    else: # assume sims is a SimFrame
        def get_timesteps(i):    
            path, time = sims.subspace.iloc[i][['path','timesteps']]
            return [os.path.join(path,f'timestep_{t:06}.vtu') for t in range(int(time))[slc]]
    def load_show_i(i):
        return loadShow(view=getview(i), files=get_timesteps(i), name=names[i], text=texts[i], field=getfield(i))
    return load_show_i


def view_simframe_slice(simframe, i, view=None, slc=None, text=None, field="spontaneous_curvature"):
    row = simframe.subspace.iloc[i]
    if slc is None:
        slc = slice(None)
    if view is None:
        view = ps.GetActiveView()
    if text is None:
        text = ", ".join(f"{k}: {row[k]:.03}" if type(row[k]) is float
                  else f'{k}:{row[k]}' for k in row.keys() if k not in ['path','store','timesteps'])
    loadShow(view, 
             [os.path.join(row.path,f'timestep_{t:06}.vtu') 
              for t in range(row.timesteps)[slc]],
             f"x{simframe.subspace.index[i]:04}", text, field=field,
             )


def get_slice_to_position_normal(s):
    """Get a source Paraview Slice"""
    return np.array(s.SliceType.Origin),np.array(s.SliceType.Normal)

def get_cylinder_distance_from_slice(s,pos,max_height=2):
    """Given source s of a Paraview Slice and v a PyVtu with pos, compute radius

    Radius of a cylinder, removing the normal component
    """
    normal = np.array(s.SliceType.Normal)
    origin = np.array(s.SliceType.Origin)
    dpos = pos-origin
    close=abs(dpos@normal)<max_height
    dpos -= (dpos@normal)[:,np.newaxis]*normal[np.newaxis,:]/(normal@normal)
    return dpos,close
