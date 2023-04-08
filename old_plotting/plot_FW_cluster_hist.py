#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script: plot a 16-block of histograms from FW files.

Created on Sun Apr 25 18:13:55 2021

@author: yoav
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import ts_vtu_to_python as v2p
import os
import concurrent.futures as cf
from itertools import product


def main():
    """Do main function.

    does the thing
    """
    main_loc = (r'/mnt/c/Users/yoavr/Desktop'
                r'/paraview_pipeline/hello_chemfarm'
                r'/pert_pan_sim_6/timesteps'
                )
    name = 'timestep'  # file name of the form {name}_f{i}w{j}_{num:06}.vtu
    step = 120
    step_range = range(step, 150)
    n_vertices = 450  # need to know this ahead to initialize
    # make 4x4 figure
    plt.rcParams['figure.figsize'] = [10, 6.8]
    # plt.rcParams['figure.dpi'] = 200
    fig, axes = plt.subplots(nrows=4, ncols=4, num="FW_6")

    all_cluster_dists = tuple(tuple(np.zeros(n_vertices+1)
                                    for x in range(4))
                              for y in range(4))

    dist_from_files(main_loc, step_range, all_cluster_dists, name)

    mean_cluster_size = [[
                            (n_vertices+1)  # dist * size = total vertices
                            / np.sum(all_cluster_dists[i][j])
                         for j in range(4)]
                         for i in range(4)]
    var_cluster_size = [[
        (np.sum(
                all_cluster_dists[i][j] *
                (range(n_vertices+1) - mean_cluster_size[i][j])**2
                )
            / np.sum(all_cluster_dists[i][j])
         )
        for j in range(4)]
        for i in range(4)]
    plot_4x4_hists(all_cluster_dists, axes,
                   mean_cluster_size, var_cluster_size)


def dist_from_files(main_loc, step_range, all_cluster_dists, name):
    """Get dist from files."""
    arg_generator = (
                     (i[0], i[1], step_range, main_loc,
                      all_cluster_dists[i[0]][i[1]], name)
                     for i in product(range(4), range(4)))
    with cf.ThreadPoolExecutor() as ex:
        ex.map(tuple_args_ij_files, arg_generator)
    # for i in range(4):
    #     for j in range(4):
    #         for step in step_range:
    #             # if there's a histogram, use it
    #             # timestep has special name rule
    #             dist_from_single_file(i, j, step,
    #                                   main_loc,
    #                                   all_cluster_dists[i][j], name)

    #         all_cluster_dists[i][j][:] /= len(step_range)
    for i in range(4):
        for j in range(4):
            all_cluster_dists[i][j][:] /= len(step_range)


def tuple_args_ij_files(arg_tuple):
    """Wrap single file."""
    dist_from_ij(*arg_tuple)


def dist_from_ij(i, j, step_range,
                 main_loc, cluster_dist, name):
    """Get cluster distribution of a single file."""
    #  start
    for step in step_range:
        if name == "timestep":
            hist_loc = f'/histogram_f{i}w{j}_{step:06}.csv'
        else:
            hist_loc = f'/histogram_{name}_f{i}w{j}_{step:06}.csv'

        if os.path.isfile(main_loc + hist_loc):  # use histogram
            ind_clust = np.genfromtxt(main_loc + hist_loc,
                                      skip_header=1,
                                      delimiter=',',
                                      dtype=int)
            cluster_dist[ind_clust[..., 0]] += ind_clust[..., 1]
        else:  # construct directly from .vtu
            specific_loc = f'/{name}_f{i}w{j}_{step:06}.vtu'
            clusdist = v2p.cluster_dist_from_vtu(main_loc
                                                 + specific_loc)
            cluster_dist[1:clusdist.shape[0]+1] += clusdist


def plot_4x4_hists(all_cluster_dists, axes,
                   mean_size, var_size):
    """Plot 16 histograms.

    plot 16 histograms in a square, in a matrixlike form
    from a 4x4 list/tuple of numpy arrays all_cluster_dists[i][j]=array([...])

    TO DO: intelligently divide into regions of 0-100 tiny clusters,
    0- 20 medium clusters, and 0-1 giant cluster
    """
    def hist_label_mean(x):
        return fr'$\left \langle N \right \rangle = {x:.2f}$'

    def hist_label_var(x):
        return fr'$\sigma = {x:.2f}$'
    # get the maximum y and x axis for the lines
    n_vtx_and_one = len(all_cluster_dists[0][0])
    max_x = np.zeros(4)
    max_y = np.zeros(4)

    for i in range(4):
        for j in range(4):
            # nonesense way to figure out "where does the distribution stops"
            # invert (trailing zeros at the start),
            # cumsum (only begining zeros survive)
            # equate 0s (1 for each trailing zeros) and sum (get how many)
            trailing_0s = (all_cluster_dists[j][i][::-1].cumsum() == 0).sum()
            max_x[j] = max(max_x[j], n_vtx_and_one - trailing_0s)
            # get "largest number of clusters"
            max_y[i] = all_cluster_dists[j][i].max()

    for i in range(3, -1, -1):
        for j in range(4):
            axe = axes[i, j]

            if max_x[j] > 100:
                axe.bar(np.arange(100), all_cluster_dists[j][i][:100],
                        width=4, color='red')
            else:
                axe.bar(np.arange(n_vtx_and_one), all_cluster_dists[j][i],
                        color='red')
            # custom legend
            axe.plot([], [], marker='.', color='red',
                     label=hist_label_mean(mean_size[j][i]))
            axe.plot([], [], marker='.', color='red',
                     label=hist_label_var(np.sqrt(var_size[j][i])))
            # plot parameters

            if i == 3:
                if max_x[j] > 100:
                    axe.set_xlim([-0.1*max_x[j], 1.1*max_x[j]])
                else:
                    axe.set_xlim([-1, 1.1*max_x[j]])
                # taken from stackoverflow: integerify the y axis
                axe.yaxis.get_major_locator().set_params(integer=True)
                # from matplotlib website
                axe.yaxis.set_minor_locator(AutoMinorLocator())
                axe.tick_params(direction='in')
            else:
                axe.sharex(axes[3, j])
                axe.tick_params(direction='in', labelbottom=False)

            if j == 0:
                axe.set_ylim([0, max_y[i]])
                # taken from stackoverflow: integerify the y axis
                axe.yaxis.get_major_locator().set_params(integer=True)
                # from matplotlib website
                axe.yaxis.set_minor_locator(AutoMinorLocator())
                axe.tick_params(direction='in')
            else:
                axe.sharey(axes[i, 0])
                axe.tick_params(direction='in', labelright=False)
            # axe.set_xlabel('cluster size')
            # axe.set_ylabel('number of clusters')
            axe.set_title(f'f{j}w{i}')
            # axe.set_aspect('equal', 'box')
            axe.legend(numpoints=1, handlelength=0,
                       markerscale=0, handletextpad=0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
