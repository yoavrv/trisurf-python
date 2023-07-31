# -*- coding: utf-8 -*-
"""Utilities."""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def nv(nshell):
    """Return number of vertices from nshell."""
    return 5*(nshell**2)+2


def _help_filish(value):
    try:
        if len(value) > 1:
            bdiff = value[1]-value[0]
            if bdiff > 1 and all(val2-val1 == bdiff
                                 for val2, val1 in zip(value[1:], value[:-1])):
                return f"_{bdiff}"
            else:
                return "__"
        else:
            return f"{value[0]:02}"
    except TypeError:  # object has no len
        return f"{value:02}"
    except Exception:
        return "__"


def filish_name(d: dict):
    """Get string representation of dict (fitting filenames)."""
    return "".join(f"{key}{_help_filish(value)}" for key, value in d.items())


def make_sidefigure(num_primary=1, ratio_p=7, ratio_q=2):
    """Make a figure with primary and secondary axes.

    fig, *axes, side_axes = make_sidefigure()
    """
    fig = plt.figure()
    gs = fig.add_gridspec(1, num_primary+1,
                          width_ratios=(*(ratio_p,)*num_primary, ratio_q),
                          # left=0.1, right=0.9, bottom=0.1, top=0.9,
                          # wspace=0.1, hspace=0.05
                          )
    axes = []
    for i in range(num_primary):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    ax_side = fig.add_subplot(gs[0, num_primary])
    ax_side.get_xaxis().set_visible(False)
    ax_side.get_yaxis().set_ticks_position('right')
    return fig, *axes, ax_side


def make_three(N, main_from=1.0, main_to=0.75, seco_from=0.0, seco_to=0.5):
    """Make a three-partition colormap [reds greens blues]."""
    cdict = {'red':   [[0.00000, seco_to, main_from],
                       [0.33333, main_to, seco_from],
                       [0.66666, seco_to, seco_from],
                       [1.00000, seco_to, seco_from]],
             'green': [[0.00000, seco_to, seco_from],
                       [0.33333, seco_to, main_from],
                       [0.66666, main_to, seco_from],
                       [1.00000, seco_to, seco_from]],
             'blue':  [[0.00000, seco_to, seco_from],
                       [0.33333, seco_to, seco_from],
                       [0.66666, seco_to, main_from],
                       [1.00000, main_to, seco_from]]}
    return LinearSegmentedColormap('testCmap', segmentdata=cdict, N=N)


def size_bar(axe, data, n=6):
    """Return linspace from x_min to x_max."""
    sizes = np.linspace(data.flatten().min(), data.flatten().max(), n)
    xmin, xmax = axe.get_xlim()
    x = np.ones(n)*(xmax-xmin)
    # y = np.linspace(*axe.get_ylim(), n+2)[1:-1]
    sc = axe.scatter(x, sizes, sizes)
    return sc, sc.values


def size_normalize(data: np.array, max_size=800, min_size=50):
    """Return data normalized for plt.scatter(s=)."""
    _copy = data.copy()
    cmin, cmax = np.nanmin(_copy), np.nanmax(_copy)
    _copy -= cmin
    _copy = _copy*(max_size-min_size)/(cmax - cmin)
    _copy = _copy + min_size
    return _copy


ColorPlotLimits = namedtuple('ColorPlotLimits',
                             ['xmin', 'xmax', 'ymin', 'ymax',
                              'datamin', 'datamax'])


def _undef_lim(x):
    if x is None:
        return np.inf, -np.inf
    else:
        return np.nanmin(x), np.nanmax(x)

def make_limits(X=None, Y=None, data=None):
    """Make a ColorPlotLimits that encompass X,Y, and data arrays."""
    return ColorPlotLimits(*_undef_lim(X), *_undef_lim(Y), *_undef_lim(data))


ColorPlotLimits.__or__ = (lambda a, b:
                          ColorPlotLimits(min(a.xmin, b.ymin),
                                          max(a.xmax, b.xmax),
                                          min(a.ymin, b.ymin),
                                          max(a.ymax, b.ymax),
                                          min(a.datamin, b.datamin),
                                          max(a.datamax, b.datamax)))


ColorPlotLimits.__and__ = (lambda a, b:
                           ColorPlotLimits(max(a.xmin, b.ymin),
                                           min(a.xmax, b.xmax),
                                           max(a.ymin, b.ymin),
                                           min(a.ymax, b.ymax),
                                           max(a.datamin, b.datamin),
                                           min(a.datamax, b.datamax)))


def color_plot(axe, X, Y, data, limits, xlabel, ylabel, title,
               sizes=None, levels=None, do_contours=False, do_contour_f=True,
               contourlevels=None, ring=False, kwargs_con=None, kwargs_scat=None):
    """Make a color plot: contour, scatter. returns sc, layers"""
    kwargs_con = {} if kwargs_con is None else kwargs_con
    kwargs_scat = {} if kwargs_scat is None else kwargs_scat
    layers = []
    if do_contour_f:
        sc = axe.contourf(X, Y, data,
                          extent=[limits.xmin, limits.xmax,
                                  limits.ymin, limits.ymax],
                          vmin=limits.datamin, vmax=limits.datamax,
                          levels=levels, **kwargs_con,
                          )
        layers.append(sc.layers)
    if do_contours or contourlevels is not None:
        if contourlevels is not None:
            levels = contourlevels
        sc = axe.contour(X, Y, data,
                    extent=[limits.xmin, limits.xmax,
                            limits.ymin, limits.ymax],
                    vmin=limits.datamin, vmax=limits.datamax,
                    levels=levels, colors='black', **kwargs_con,)
        layers.append(sc.layers)
    if sizes is not None:
        if ring:
            sc = axe.scatter(X, Y, sizes, data,
                             vmin=limits.datamin, vmax=limits.datamax,
                             edgecolor=[0, 0, 0, 0.5], linewidth=0.25, **kwargs_scat,)

        else:
            sc = axe.scatter(X, Y, sizes, data,
                             vmin=limits.datamin, vmax=limits.datamax, **kwargs_scat,)
            # more colorful

    axe.set_xlim(limits.xmin, limits.xmax)
    axe.set_ylim(limits.ymin, limits.ymax)
    if xlabel is not None:
        axe.set_xlabel(xlabel)
    if ylabel is not None:
        axe.set_ylabel(ylabel)
    if title is not None:
        axe.set_title(title)
    return sc, layers
