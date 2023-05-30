#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:03:01 2022.

fitting of sphere by fibonacci sequence, based on a blog post:
http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/


@author: yoav
"""
import sys
sys.path.append("/opt/workspace/msc_project/lab_note_and_py/trisurf-python/")

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from PyVtu_plot import (setupAxe3D, plotMySurf,
                        points_filter, blist_to_flatnan,
                        unique1D_array_from_list)

# %% fibonacci definitions


def epsilon_offset(n):
    """Generate offset epsilon value from table, maximizing minimum distance.

    copied from the blog post
    """
    if n >= 600_000:
        return 214
    elif n >= 400_000:
        return 75
    elif n >= 11_000:
        return 27
    elif n >= 890:
        return 10
    elif n >= 177:
        return 3.33
    elif n >= 24:
        return 1.33
    else:
        return 0.33


def epsilon_average(n):
    """Generate offset epsilon value that maximizes the average distance.

    see the blog
    """
    return 0.36

# %% main function for running the fibonacci demonstration


def fibo_main(n):
    """Create an n-vertex triangulation of a sphere.

    partially copied from the blog post
    """
    epsilon = epsilon_average(n)
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio  # I hate this inverted phi/theta
    phi = np.arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    fig, axe = setupAxe3D(zoom=0.95)
    axe.plot3D(x, y, z, '.k', markersize=0.4)

    points = np.vstack((x, y, z)).T
    # add a center point for the triangulation:
    points2 = np.vstack(([0, 0, 0], points))
    tri = Delaunay(points2)  # tetrahedrals of center vtx (+) surface triangle
    edges = []
    cells = []
    for simp in tri.simplices:
        a, b, c = simp[simp != 0]-1
        edges.extend(([a, b], [b, c], [c, a]))
        cells.append([a, b, c])
    edges = unique1D_array_from_list(edges)
    cells = unique1D_array_from_list(cells)
    xe, ye, ze = points[edges, :].T
    xen, yen, zen = [blist_to_flatnan(x) for x in (xe, ye, ze)]
    axe.plot3D(xen, yen, zen, linewidth=0.5)
    fig2, axe2 = setupAxe3D(aspect=(1, 1, 1), zoom=0.95)
    eighth = (z <= 0) & ((x <= 0) & (y <= 0))
    p, b, _ = points_filter(eighth, points, edges, cells)
    plotMySurf(axe2, points=p, bonds=b)
    plt.tight_layout()
    return points, edges, cells, axe, axe2


def fibo_main2(n):
    """Create an n-vertex triangulation of a sphere.

    partially copied from the blog post
    """
    epsilon = epsilon_average(n)
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio  # I hate this inverted phi/theta
    phi = np.arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    points = np.vstack((x, y, z)).T
    # add a center point for the triangulation:
    points2 = np.vstack(([0, 0, 0], points))
    tri = Delaunay(points2)  # tetrahedrals of center vtx (+) surface triangle
    edges = []
    cells = []
    for simp in tri.simplices:
        a, b, c = simp[simp != 0]-1
        edges.extend(([a, b], [b, c], [c, a]))
        cells.append([a, b, c])
    edges = unique1D_array_from_list(edges)
    cells = unique1D_array_from_list(cells)
    # in __name__==__main__:
    # res = main(n=2000)
    # points, edges, cells, axe, axe2 = res
    # h = [main2(int(x)) for x in np.geomspace(60, 2500, 20)]
    # # same points/bonds,triangle number as the bipyramid!
    return points, edges, cells


if __name__ == '__main__':
    fibo_main(n=500)
