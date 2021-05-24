#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script: make fxw plot of distribution from FW summaries.

Created on Sun May 23 2021

@author: yoav
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures as cf
from itertools import product


def main():
    """Do main function.

    does the thing
    """
    main_loc = (r'/mnt/c/Users/yoavr/Desktop'
                r'/paraview_pipeline/hello_chemfarm'
                r'/FW_block_aggregated/timesteps'
                )
    name = "std_cluster_size"

    frange = range(2, 4)  # range f
    wrange = range(5, 9)
    ifrange = range(len(frange))  # index range
    iwrange = range(len(wrange))

    step_range = range(0, 150)  # step range

    dists = tuple(tuple(np.zeros(len(step_range))
                        for x in iwrange)
                  for y in ifrange)

    dist_from_files(main_loc, frange, wrange, step_range,
                    dists, name)

    mean_dist = [[dists[i][j].mean()
                  for j in iwrange]
                 for i in ifrange]
    var_dist = [[dists[i][j].var()
                 for j in iwrange]
                for i in ifrange]

    plot_fxw_hists(dists, frange, wrange, step_range,
                   mean_dist, var_dist, name)


def dist_from_files(main_loc, frange, wrange, step_range,
                    dists, name):
    """Get dist from files."""
    ifrange = range(len(frange))
    iwrange = range(len(wrange))
    arg_generator = (
                     (frange[i[0]], wrange[i[1]], step_range, main_loc,
                      dists[i[0]][i[1]], name)
                     for i in product(ifrange, iwrange)
                    )
    with cf.ThreadPoolExecutor() as ex:
        ex.map(tuple_args_ij_files, arg_generator)

    #         all_cluster_dists[i][j][:] /= len(step_range)
    # for i in ifrange:
    #     for j in iwrange:
    #         dists[i][j][:] /= len(step_range)


def tuple_args_ij_files(arg_tuple):
    """Wrap single file."""
    dist_from_ij(*arg_tuple)


def dist_from_ij(f, w, step_range,
                 main_loc, dist, name):
    """Get cluster distribution of a single file."""
    #  start
    dist_loc = f'/f{f}w{w}/main_statistics.csv'
    df = pd.read_csv(main_loc + dist_loc)
    for k, step in enumerate(step_range):
        dist[k] = df[name][step]


def plot_fxw_hists(dists, frange, wrange, step_range,
                   mean_dist, var_dist, name, do_legend=True, do_title=True):
    """Plot fxw graphs.

    plot fxw graphs in a square, in a matrixlike form
    from a fxw list/tuple of numpy arrays dists[i][j]=array([...])
    """
    def dist_label_mean(x):
        return fr'$\mu = {x:.2f}$'

    def dist_label_var(x):
        return fr'$\sigma = {x:.2f}$'

    flen = len(frange)
    wlen = len(wrange)
    ifrange = range(flen)
    iwrange = range(wlen)

    # get the range of y
    max_y = np.zeros(wlen)
    min_y = np.zeros(wlen)

    # f is in the 2nd, horizontal axis
    # w is in the 1st, vertical axis
    for i in iwrange:
        for j in ifrange:
            # get maximum hieght
            max_y[i] = dists[j][i].max()
            min_y[i] = dists[j][i].min()

    # make big figure
    plt.rcParams['figure.figsize'] = [10, 6.8]
    # plt.rcParams['figure.dpi'] = 200
    fig, axes = plt.subplots(nrows=wlen, ncols=flen, num=name)

    # do axis things on the axis
    bottom_y = len(iwrange)-1
    left_x = 0

    for i, w in zip(reversed(iwrange), reversed(wrange)):
        for j, f in zip(ifrange, frange):
            axe = axes[i, j]

            axe.plot(step_range, dists[j][i],
                     color='red', linewidth=0.2)
            if do_legend:
                # custom legend
                axe.plot([], [], marker='.', color='red',
                         label=dist_label_mean(mean_dist[j][i]))
                axe.plot([], [], marker='.', color='red',
                         label=dist_label_var(np.sqrt(var_dist[j][i])))
            # plot parameters

            if i == bottom_y:
                5
                #axe.tick_params(direction='in')
            else:
                #axe.sharex(axes[bottom_y, j])
                axe.tick_params(direction='in', labelbottom=False)

            if j == left_x:
                #axe.set_ylim([min_y[i], 1.1*max_y[i]])
                axe.tick_params(direction='in')
            else:
                #axe.sharey(axes[i, left_x])
                axe.tick_params(direction='in', labelright=False)
            # axe.set_xlabel('cluster size')
            # axe.set_ylabel('number of clusters')
            if do_title:
                axe.set_title(f'f=0.{f}, w={0.5+0.25*w}: {name}')
            # axe.set_aspect('equal', 'box')
            if do_legend:
                axe.legend(numpoints=1, handlelength=0,
                           markerscale=0, handletextpad=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
