#!/usr/bin/env python
# coding: utf-8
"""Get statistics on a series of vtu files.

output a statistics file, optionally write cluster size histogram
"""

# imports: should be in standard anaconda distribution
import xml.etree.ElementTree as ET
import os
import argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
import csv
from numba import njit
#from numba.typed import List as nb_List
# "upside down": main is at the bottom, major function get_statistics above
# that, extra helper function at top"

# xml functions, jitted functions, main numeric function, main function

# xml functions: essentially a new vtu_to_python:


def xml_to_np(node, dtype=np.dtype(float)):
    r"""Take xml node, extract text array to numpy array of dtype.

    Convert sequence to 1d array and lines to 2d arrays, e.g
     2 3\n4 5 6\n' -> [['2','3'],['4','5','6']]
    '\n 2 3 5' -> ['2','3','5']
    """
    all_text = node.text.strip()

    # if there are tabs in the middle, assume this is a 2d list
    if all_text.__contains__('\n'):
        return np.array([x.split() for x in all_text.split('\n')], dtype=dtype)
    else:
        return np.array((all_text.split()), dtype=dtype)


def xml_to_2_np(node, dtype=np.dtype(int)):
    r"""Take xml node, extract text array to two numpy arrays.

    Convert lines to 2d arrays, e.g
     2 3\n4 5 6\n' -> [['2','3'],['4','5','6']]
    '\n 2 3 5' -> ['2','3','5']
    split based on their length
    """
    all_text = node.text.strip()

    # We assume this is a 2D array

    tstring = [x.split() for x in all_text.split('\n')]

    return (np.array([a for a in tstring if len(a) == 2], dtype=dtype),
            np.array([a for a in tstring if len(a) == 3], dtype=dtype))


def np_to_xml(node, data_array):
    r"""Insert numpy array as text array of node.

    reverse of xml_to_np
    """
    # maintain the current format (pre- and -post spaces and tabs)
    pre, post = '', ''
    for i in node.text:
        if i == ' ' or i == '\n':
            pre = pre + i
        else:
            break
    for i in reversed(node.text):
        if i == ' ' or i == '\n':
            post = i + post
        else:
            break

    # choose number format based on array type
    if data_array.dtype == np.float64:
        def fmt(x): return f"{x:.17e}"
    else:
        def fmt(x): return f"{x}"

    # insert numpy array as text: dimension matters!
    if data_array.ndim == 0:
        node.text = pre + fmt(data_array) + post
    if data_array.ndim == 1:
        node.text = pre + " ".join(fmt(x) for x in data_array) + post
    if data_array.ndim == 2:
        node.text = (pre
                     + "\n".join("".join(fmt(x) for x in y)
                                 for y in data_array)
                     + post)


# njitted functions

@njit
def numpy_sum_extend(array_to_add_to, array_extend_indices, array_to_add_from):
    """Apply A[B] += D, even when B and D are larger than A."""
    for i, j in enumerate(array_extend_indices):
        array_to_add_to[j, ...] += array_to_add_from[i, ...]


@njit
def strong_connect(i, c_idx, S, CC, v_idx, v_lowlink,
                   v_on_stack, blist, v_keep=None):
    """Wikicopy."""
    v_idx[i] = c_idx+0
    v_lowlink[i] = c_idx+0
    c_idx += 1
    S.append(i)
    v_on_stack[i] = True
    for bond in blist:
        if bond[0] == i:
            j = bond[1]
        elif bond[1] == i:
            j = bond[0]
        else:
            continue
        if v_keep is not None and not v_keep[j]:
            continue
        #
        if v_idx[j] == 0:
            strong_connect(j, c_idx, S, CC,
                           v_idx, v_lowlink,  v_on_stack, blist, v_keep)
            v_lowlink[i] = min(v_lowlink[i], v_lowlink[j])
        elif v_on_stack[j]:
            v_lowlink[i] = min(v_lowlink[i], v_idx[j])

    # if v is root:
    if v_idx[i] == v_lowlink[i]:
        currCC = []  # new component
        while True:
            j = S.pop()
            v_on_stack[j] = False
            v_lowlink[j] = v_lowlink[i]
            currCC.append(j)
            if j == i:
                break
        CC.append(currCC)


@njit
def connected_components(nvtx, blist, v_keep=None):
    """Get connected components for nvtx nodes in v_keep; connected by blist.

    vtx are assumed to be [0...nvtx-1]
    Mostly copied from Wikipedia
    "Trajan's strongly connected components algorithm"
    (which can't be too good: we're not having any strong connection)

    returns list of list of nodes i.e. list of clusters and a cluster_id array
    """
    v_idx = np.zeros(nvtx, dtype=np.int64)
    v_lowlink = np.zeros(nvtx, dtype=np.int64)
    v_on_stack = np.zeros(nvtx, dtype=np.bool_)
    c_idx = np.array(1)  # current index
    S = []
    S.append(1)
    S.pop()
    CC = []
    CC.append(S)
    CC.pop()
    for i in range(len(v_idx)):
        if v_keep is not None and not v_keep[i]:
            continue
        if v_idx[i] == 0:
            strong_connect(i, c_idx, S, CC, v_idx, v_lowlink,
                           v_on_stack, blist, v_keep)
    for i, clst in enumerate(CC):
        for node in clst:
            v_lowlink[node] = i
    return CC, v_lowlink


@njit
def calculate_statistic_new_w(node_type, c, pos, blist, tlist,
                              bending_E):
    """Calculate the statistics from the arrays."""
    """Calculate the statistics from the arrays."""
    nvtx = len(node_type)
    bonding = node_type & 1 != 0

    t_normals = np.zeros((tlist.shape[0], 3))

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    ######################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    ##############################################################
    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    ######################################################
    # mean curvature:
    # new version saves it, but we need sigmas for the perimeters
    # a lot harder, since we don't have the neighbors directly.
    # the components of summation are,
    # for each vertex i:
    #   sum all l_ij * cotan(theta_opposite)/2 --> rh[i]
    #   sum normal of triangles (to determine h sign) --> tnh[i]
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # add the normal to each vertex in the triangle:
    # vtx_normal[tri->vtx[0]] += tri->normal. then for 1 and 2
    # problematic due to repeated indices in triangles- two triangles can
    # have the same vertex in 0, screwing the +=
    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)
    # we only need direction, tnh*rh<0, so no need to normalize

    # Summing the other part is more difficult
    # we go on each vertex of the triangle k=[0,1,2]
    # calculate cotan(theta[k])
    # and add the relevant lij*cotan(theta[k])/2 vector to rh[i!=k]

    # To get cotan, we will beed bond length square
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    # numpy vectorized version of the c calculation
    # cot[q] = |a||b|cos/sqrt(|a|^2|b|^2 - |a|^2|b|^2cos^2)
    # |a||b|cos = a @ b
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    # contributions to 1 and 2: +-l_12 * cot(theta[0])=+-sigma12
    # (divide by 2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    # few! that was not nice

    ####################################
    # cluster size distribution:

    if not bonding.any() and (node_type & 2).any():
        CC, labeled_vtx = connected_components(nvtx, blist, (node_type & 2))
    else:
        CC, labeled_vtx = connected_components(nvtx, blist, bonding)

    n_clusters = len(CC)
    if n_clusters == 0:
        mean_cluster_size = 0
        std_cluster_size = np.nan
        perim = 0
    else:
        labeled_vtx[~bonding] = n_clusters
        mean_cluster_size = 0.
        std_cluster_size = 0.
        for clst in CC:
            mean_cluster_size += len(clst)
            mean_cluster_size /= n_clusters
        for clst in CC:
            std_cluster_size += (len(clst)-mean_cluster_size)**2
        std_cluster_size /= n_clusters-1
        std_cluster_size = np.sqrt(std_cluster_size)

        # horrifying monstrosity: associate every vertex with perimeter
        # of the non-clustered neighbors
        # correct sigmas:
        sig_12 = np.sqrt((sigma_12**2).sum(axis=-1))/2
        sig_20 = np.sqrt((sigma_20**2).sum(axis=-1))/2
        sig_01 = np.sqrt((sigma_01**2).sum(axis=-1))/2
        vtx_perim = np.zeros(nvtx, dtype=np.float64)
        add_to_1_2 = labeled_vtx[tlist[:, 1]] != labeled_vtx[tlist[:, 2]]
        numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 1],
                         sig_12[add_to_1_2])
        numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 2],
                         sig_12[add_to_1_2])
        add_to_2_0 = labeled_vtx[tlist[:, 2]] != labeled_vtx[tlist[:, 0]]
        numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 2],
                         sig_20[add_to_2_0])
        numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 0],
                         sig_20[add_to_2_0])
        add_to_0_1 = labeled_vtx[tlist[:, 0]] != labeled_vtx[tlist[:, 1]]
        numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 0],
                         sig_01[add_to_0_1])
        numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 1],
                         sig_01[add_to_0_1])

        perim = vtx_perim[bonding].sum()

    ##############################
    # get statistics of individual clusters
    # stats: size, perim, l, E
    clst_size = np.zeros(n_clusters, dtype=np.int64)
    clst_perim = np.zeros(n_clusters, dtype=np.float64)
    lam = np.zeros((n_clusters, 3), dtype=np.float64)
    clst_E = np.zeros(n_clusters, dtype=np.float64)
    clst_sample_id = np.zeros(n_clusters, dtype=np.int64)

    for i, clst_list in enumerate(CC):
        clst = np.array(clst_list)
        clst_size[i] = len(clst)
        clst_E[i] = bending_E[clst].sum()
        clst_gy_eig = np.linalg.eigvalsh(
            (pos[clst].T @ pos[clst]) / clst_size[i])
        clst_gy_eig.sort()
        lam[i, :] = clst_gy_eig
        clst_perim[i] = vtx_perim[clst].sum()
        clst_sample_id[i] = clst[0]

    return (
            (total_volume, total_area,
             gy_eig[0], gy_eig[1], gy_eig[2],
             nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size),
            (clst_size, clst_E, lam[:, 0], lam[:, 1], lam[:, 2], clst_perim,
             clst_sample_id)
            )


@njit
def calculate_statistic_new_ww(node_type, c, pos, blist, tlist,
                               bending_E):
    """Calculate the statistics from the arrays."""
    """Calculate the statistics from the arrays."""
    nvtx = len(node_type)
    bonding = node_type & 1 != 0

    t_normals = np.zeros((tlist.shape[0], 3))

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    ######################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    ##############################################################
    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    ######################################################
    # mean curvature:
    # new version saves it, but we need sigmas for the perimeters
    # a lot harder, since we don't have the neighbors directly.
    # the components of summation are,
    # for each vertex i:
    #   sum all l_ij * cotan(theta_opposite)/2 --> rh[i]
    #   sum normal of triangles (to determine h sign) --> tnh[i]
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # add the normal to each vertex in the triangle:
    # vtx_normal[tri->vtx[0]] += tri->normal. then for 1 and 2
    # problematic due to repeated indices in triangles- two triangles can
    # have the same vertex in 0, screwing the +=
    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)
    # we only need direction, tnh*rh<0, so no need to normalize

    # Summing the other part is more difficult
    # we go on each vertex of the triangle k=[0,1,2]
    # calculate cotan(theta[k])
    # and add the relevant lij*cotan(theta[k])/2 vector to rh[i!=k]

    # To get cotan, we will beed bond length square
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    # numpy vectorized version of the c calculation
    # cot[q] = |a||b|cos/sqrt(|a|^2|b|^2 - |a|^2|b|^2cos^2)
    # |a||b|cos = a @ b
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    # contributions to 1 and 2: +-l_12 * cot(theta[0])=+-sigma12
    # (divide by 2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    # few! that was not nice

    ####################################
    # cluster size distribution:

    if not bonding.any() and (node_type & 2).any():
        CC, labeled_vtx = connected_components(nvtx, blist, (node_type & 2))
    else:
        CC, labeled_vtx = connected_components(nvtx, blist, bonding)

    n_clusters = len(CC)
    if n_clusters == 0:
        mean_cluster_size = 0
        std_cluster_size = np.nan
        perim = 0
    else:
        labeled_vtx[~bonding] = n_clusters
        mean_cluster_size = 0.
        std_cluster_size = 0.
        for clst in CC:
            mean_cluster_size += len(clst)
            mean_cluster_size /= n_clusters
        for clst in CC:
            std_cluster_size += (len(clst)-mean_cluster_size)**2
        std_cluster_size /= n_clusters-1
        std_cluster_size = np.sqrt(std_cluster_size)

        # horrifying monstrosity: associate every vertex with perimeter
        # of the non-clustered neighbors
        # correct sigmas:
        sig_12 = np.sqrt((sigma_12**2).sum(axis=-1))/2
        sig_20 = np.sqrt((sigma_20**2).sum(axis=-1))/2
        sig_01 = np.sqrt((sigma_01**2).sum(axis=-1))/2
        vtx_perim = np.zeros(nvtx, dtype=np.float64)
        add_to_1_2 = labeled_vtx[tlist[:, 1]] != labeled_vtx[tlist[:, 2]]
        numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 1],
                         sig_12[add_to_1_2])
        numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 2],
                         sig_12[add_to_1_2])
        add_to_2_0 = labeled_vtx[tlist[:, 2]] != labeled_vtx[tlist[:, 0]]
        numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 2],
                         sig_20[add_to_2_0])
        numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 0],
                         sig_20[add_to_2_0])
        add_to_0_1 = labeled_vtx[tlist[:, 0]] != labeled_vtx[tlist[:, 1]]
        numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 0],
                         sig_01[add_to_0_1])
        numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 1],
                         sig_01[add_to_0_1])

        perim = vtx_perim[bonding].sum()

    ##############################
    # get statistics of individual clusters
    # stats: size, perim, l, E
    clst_size = np.zeros(n_clusters, dtype=np.int64)
    clst_perim = np.zeros(n_clusters, dtype=np.float64)
    lam = np.zeros((n_clusters, 3), dtype=np.float64)
    clst_E = np.zeros(n_clusters, dtype=np.float64)
    clst_sample_id = np.zeros(n_clusters, dtype=np.int64)

    for i, clst_list in enumerate(CC):
        clst = np.array(clst_list)
        clst_size[i] = len(clst)
        clst_E[i] = bending_E[clst].sum()
        clst_gy_eig = np.linalg.eigvalsh(
            (pos[clst].T @ pos[clst]) / clst_size[i])
        clst_gy_eig.sort()
        lam[i, :] = clst_gy_eig
        clst_perim[i] = vtx_perim[clst].sum()
        clst_sample_id[i] = clst[0]

    #######################
    # typed clusters:
    type_blist = blist[node_type[blist[:, 0]] == node_type[blist[:, 1]], :]
    CC2, labeled_vtx = connected_components(nvtx, type_blist)
    n_clusters = len(CC2)
    ##############################
    # get statistics of individual clusters
    # stats: size, perim, l, E
    clst_type = np.zeros(n_clusters, dtype=np.int64)
    clst_size_typed = np.zeros(n_clusters, dtype=np.int64)
    clst_perim_typed = np.zeros(n_clusters, dtype=np.float64)
    lam_typed = np.zeros((n_clusters, 3), dtype=np.float64)
    clst_E_typed = np.zeros(n_clusters, dtype=np.float64)
    clst_sample_id_typed = np.zeros(n_clusters, dtype=np.int64)

    ############################################################
    # horrifying monstrosity: associate every vertex with perimeter
    # of the non-clustered neighbors. now with types!
    # correct sigmas:
    # sig_12 = np.sqrt((sigma_12**2).sum(axis=-1))/2
    # sig_20 = np.sqrt((sigma_20**2).sum(axis=-1))/2
    # sig_01 = np.sqrt((sigma_01**2).sum(axis=-1))/2
    vtx_perim = np.zeros(nvtx, dtype=np.float64)
    add_to_1_2 = labeled_vtx[tlist[:, 1]] != labeled_vtx[tlist[:, 2]]
    numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 1],
                     sig_12[add_to_1_2])
    numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 2],
                     sig_12[add_to_1_2])
    add_to_2_0 = labeled_vtx[tlist[:, 2]] != labeled_vtx[tlist[:, 0]]
    numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 2],
                     sig_20[add_to_2_0])
    numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 0],
                     sig_20[add_to_2_0])
    add_to_0_1 = labeled_vtx[tlist[:, 0]] != labeled_vtx[tlist[:, 1]]
    numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 0],
                     sig_01[add_to_0_1])
    numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 1],
                     sig_01[add_to_0_1])

    for i, clst_list in enumerate(CC2):
        clst = np.array(clst_list)
        clst_size_typed[i] = len(clst)
        clst_E_typed[i] = bending_E[clst].sum()
        clst_gy_eig = np.linalg.eigvalsh(
            (pos[clst].T @ pos[clst]) / clst_size_typed[i])
        clst_gy_eig.sort()
        lam_typed[i, :] = clst_gy_eig
        clst_perim_typed[i] = vtx_perim[clst].sum()
        clst_sample_id_typed[i] = clst[0]
        clst_type[i] = node_type[clst[0]]

    return (
            (total_volume, total_area,
             gy_eig[0], gy_eig[1], gy_eig[2],
             nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size),
            (clst_size, clst_E, lam[:, 0], lam[:, 1], lam[:, 2], clst_perim,
             clst_sample_id), (clst_type, clst_size_typed, clst_E_typed,
                               lam_typed[:, 0], lam_typed[:, 1],
                               lam_typed[:, 2], clst_perim_typed,
                               clst_sample_id_typed)
            )


@njit
def calculate_statistic_new(node_type, c, pos, blist, tlist,
                            bending_E):
    """Calculate the statistics from the arrays."""
    """Calculate the statistics from the arrays."""
    nvtx = len(node_type)
    bonding = node_type & 1 != 0

    t_normals = np.zeros((tlist.shape[0], 3))

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    ######################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    ##############################################################
    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    ######################################################
    # mean curvature:
    # new version saves it, but we need sigmas for the perimeters
    # a lot harder, since we don't have the neighbors directly.
    # the components of summation are,
    # for each vertex i:
    #   sum all l_ij * cotan(theta_opposite)/2 --> rh[i]
    #   sum normal of triangles (to determine h sign) --> tnh[i]
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # add the normal to each vertex in the triangle:
    # vtx_normal[tri->vtx[0]] += tri->normal. then for 1 and 2
    # problematic due to repeated indices in triangles- two triangles can
    # have the same vertex in 0, screwing the +=
    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)
    # we only need direction, tnh*rh<0, so no need to normalize

    # Summing the other part is more difficult
    # we go on each vertex of the triangle k=[0,1,2]
    # calculate cotan(theta[k])
    # and add the relevant lij*cotan(theta[k])/2 vector to rh[i!=k]

    # To get cotan, we will beed bond length square
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    # numpy vectorized version of the c calculation
    # cot[q] = |a||b|cos/sqrt(|a|^2|b|^2 - |a|^2|b|^2cos^2)
    # |a||b|cos = a @ b
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    # contributions to 1 and 2: +-l_12 * cot(theta[0])=+-sigma12
    # (divide by 2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    # few! that was not nice

    ####################################
    # cluster size distribution:
    if not bonding.any() and (node_type & 2).any():
        CC, labeled_vtx = connected_components(nvtx, blist, (node_type & 2))
    else:
        CC, labeled_vtx = connected_components(nvtx, blist, bonding)

    n_clusters = len(CC)
    if n_clusters == 0:
        mean_cluster_size = 0
        std_cluster_size = np.nan
        perim = 0
    else:
        labeled_vtx[~bonding] = n_clusters
        mean_cluster_size = 0.
        std_cluster_size = 0.
        for clst in CC:
            mean_cluster_size += len(clst)
            mean_cluster_size /= n_clusters
        for clst in CC:
            std_cluster_size += (len(clst)-mean_cluster_size)**2
        std_cluster_size /= n_clusters-1
        std_cluster_size = np.sqrt(std_cluster_size)

        # horrifying monstrosity: associate every vertex with perimeter
        # of the non-clustered neighbors
        # correct sigmas:
        sig_12 = np.sqrt((sigma_12**2).sum(axis=-1))/2
        sig_20 = np.sqrt((sigma_20**2).sum(axis=-1))/2
        sig_01 = np.sqrt((sigma_01**2).sum(axis=-1))/2
        vtx_perim = np.zeros(nvtx, dtype=np.float64)
        add_to_1_2 = labeled_vtx[tlist[:, 1]] != labeled_vtx[tlist[:, 2]]
        numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 1],
                         sig_12[add_to_1_2])
        numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 2],
                         sig_12[add_to_1_2])
        add_to_2_0 = labeled_vtx[tlist[:, 2]] != labeled_vtx[tlist[:, 0]]
        numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 2],
                         sig_20[add_to_2_0])
        numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 0],
                         sig_20[add_to_2_0])
        add_to_0_1 = labeled_vtx[tlist[:, 0]] != labeled_vtx[tlist[:, 1]]
        numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 0],
                         sig_01[add_to_0_1])
        numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 1],
                         sig_01[add_to_0_1])

        perim = vtx_perim[bonding].sum()

    return (total_volume, total_area,
            gy_eig[0], gy_eig[1], gy_eig[2],
            nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size
            )


@njit
def calculate_statistic_new_dep(nodetype, c, v_normals, pos,
                                blist, tlist, v=False, w=False):
    """Calculate the statistics from the arrays."""
    nvtx = len(nodetype)
    bonding = nodetype & 1 != 0

    t_normals = np.zeros((tlist.shape[0], 3))

    ######################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.linalg.norm(t_normals, axis=1)
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    eighteen_volumes = np.einsum('ij,ij->i', (xyz0 + xyz1 + xyz2), t_normals)
    total_area = double_areas.sum()/2
    total_volume = -eighteen_volumes.sum()/18

    ##################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    ##############################################################
    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    ######################################################
    # mean curvature:
    # new version saves it, but we need sigmas for the perimeters
    # we have this already
    hmean = sum(c)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= double_areas[:, np.newaxis]  # normalizing vectors was skipped

    # we have the normals saved, but we still need stuff for the perimeters
    # tnh = v_normals

    #############################################################
    # perimeter: if vertex "i" in a triangle is unique, there's a
    # domain boundary running through: |sigma_ij|+|sigma_ik|
    perim = 0

    # we need bond length square
    bond_sqr01 = np.einsum('ij,ij->i', xyz1-xyz0, xyz1-xyz0)
    bond_sqr02 = np.einsum('ij,ij->i', xyz2-xyz0, xyz2-xyz0)
    bond_sqr12 = np.einsum('ij,ij->i', xyz2-xyz1, xyz2-xyz1)

    # on 0th vtx of each triangle:
    # numpy vectorized version of the c calculation
    # cot[q] = |a||b|cos/sqrt(|a|^2|b|^2 - |a|^2|b|^2cos^2)
    # |a||b|cos = a @ b
    dot_prod_at = np.einsum('ij,ij->i', xyz1-xyz0, xyz2-xyz0)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = cot_at[:, np.newaxis] * (xyz2 - xyz1)

    # on 1th vtx of each triangle
    dot_prod_at = np.einsum('ij,ij->i', xyz2-xyz1, xyz0-xyz1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = cot_at[:, np.newaxis] * (xyz0 - xyz2)

    # on 2th vtx
    dot_prod_at = np.einsum('ij,ij->i', xyz0-xyz2, xyz1-xyz2)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = cot_at[:, np.newaxis] * (xyz1 - xyz0)

    # perimeters:
    # for any 0th vertex that is unique, 0!=1 and 1==2
    unique_vtx = ((bonding[tlist[:, 0]] != bonding[tlist[:, 1]])
                  & (bonding[tlist[:, 1]] == bonding[tlist[:, 2]]))
    # += |sigma_20|
    perim += np.linalg.norm(sigma_20[unique_vtx, :], axis=1).sum()
    # += |sigma_01|
    perim += np.linalg.norm(sigma_01[unique_vtx, :], axis=1).sum()
    # same for any 1th vertex that is unique, 1!=2 and 2==0
    unique_vtx = ((bonding[tlist[:, 1]] != bonding[tlist[:, 2]])
                  & (bonding[tlist[:, 2]] == bonding[tlist[:, 0]]))
    perim += np.linalg.norm(sigma_01[unique_vtx, :], axis=1).sum()
    perim += np.linalg.norm(sigma_12[unique_vtx, :], axis=1).sum()
    # same for any 2th vertex that is unique, 2!=0 and 0==1
    unique_vtx = ((bonding[tlist[:, 2]] != bonding[tlist[:, 0]])
                  & (bonding[tlist[:, 0]] == bonding[tlist[:, 1]]))
    perim += np.linalg.norm(sigma_12[unique_vtx, :], axis=1).sum()
    perim += np.linalg.norm(sigma_20[unique_vtx, :], axis=1).sum()
    # sigmas are still 2*sigma
    perim /= 2

    ####################################
    # cluster size distribution:
    # rehash of what's done in ts_vtu_to_python
    # but using more obtuse numpy trick
    adj = sp.lil_matrix((nvtx, nvtx), dtype=bool)
    clst_bonds = bonding[blist[:, 0]] & bonding[blist[:, 1]]
    adj[blist[clst_bonds, 0], blist[clst_bonds, 1]] = True
    _, labeled_vtx = sp.csgraph.connected_components(adj, directed=False)
    dist_size = np.bincount(np.bincount(labeled_vtx[bonding]))[1:]
    n_clusters = dist_size.sum()
    mean_cluster_size = (dist_size @ range(1, dist_size.size+1)
                         / n_clusters)
    std_cluster_size = (dist_size @
                        (range(1, dist_size.size+1)-mean_cluster_size)**2
                        / (n_clusters-1))
    std_cluster_size = np.sqrt(std_cluster_size)

    ##############################
    # write histogram if requested
    if w:
        # v2p.write_cluster_hist(dist_size, vtu_location, v)
        # to redo
        if w == 1:
            pass
        if w == 2:
            pass

    return (total_volume, total_area,
            gy_eig[0], gy_eig[1], gy_eig[2],
            nbw_nb, hmean, mean_cluster_size, std_cluster_size, perim)


@njit
def calculate_statistic_old(active, pos, blist, tlist):
    """Calculate the statistics from the arrays."""
    nvtx = len(active)

    t_normals = np.zeros((tlist.shape[0], 3))

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    ######################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    ##############################################################
    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (active[blist[:, 0]] & active[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    ######################################################
    # mean curvature:
    # new version saves it, but we need sigmas for the perimeters
    # a lot harder, since we don't have the neighbors directly.
    # the components of summation are,
    # for each vertex i:
    #   sum all l_ij * cotan(theta_opposite)/2 --> rh[i]
    #   sum normal of triangles (to determine h sign) --> tnh[i]
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # add the normal to each vertex in the triangle:
    # vtx_normal[tri->vtx[0]] += tri->normal. then for 1 and 2
    # problematic due to repeated indices in triangles- two triangles can
    # have the same vertex in 0, screwing the +=
    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)
    # we only need direction, tnh*rh<0, so no need to normalize

    # Summing the other part is more difficult
    # we go on each vertex of the triangle k=[0,1,2]
    # calculate cotan(theta[k])
    # and add the relevant lij*cotan(theta[k])/2 vector to rh[i!=k]

    # To get cotan, we will beed bond length square
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    # numpy vectorized version of the c calculation
    # cot[q] = |a||b|cos/sqrt(|a|^2|b|^2 - |a|^2|b|^2cos^2)
    # |a||b|cos = a @ b
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    # contributions to 1 and 2: +-l_12 * cot(theta[0])=+-sigma12
    # (divide by 2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    # few! that was not nice

    #############################################################
    # perimeter: if vertex "i" in a triangle is unique, there's a
    # domain boundary running through: |sigma_ij|+|sigma_ik|
    # perim = 0
    # # for any 0th vertex that is unique, 0!=1 and 1==2
    # unique_vtx = ((active[tlist[:, 0]] != active[tlist[:, 1]])
    #               & (active[tlist[:, 1]] == active[tlist[:, 2]]))
    # # += |sigma_20|
    # perim += np.sqrt((sigma_20[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # += |sigma_01|
    # perim += np.sqrt((sigma_01[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # same for any 1th vertex that is unique, 1!=2 and 2==0
    # unique_vtx = ((active[tlist[:, 1]] != active[tlist[:, 2]])
    #               & (active[tlist[:, 2]] == active[tlist[:, 0]]))
    # perim += np.sqrt((sigma_01[unique_vtx, :]**2).sum(axis=-1)).sum()
    # perim += np.sqrt((sigma_12[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # same for any 2th vertex that is unique, 2!=0 and 0==1
    # unique_vtx = ((active[tlist[:, 2]] != active[tlist[:, 0]])
    #               & (active[tlist[:, 0]] == active[tlist[:, 1]]))
    # perim += np.sqrt((sigma_12[unique_vtx, :]**2).sum(axis=-1)).sum()
    # perim += np.sqrt((sigma_20[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # sigmas are still 2*sigma
    # perim /= 2

    ####################################
    # cluster size distribution:

    CC, labeled_vtx = connected_components(nvtx, blist, active)
    n_clusters = len(CC)
    labeled_vtx[~active] = n_clusters
    mean_cluster_size = 0.
    std_cluster_size = 0.
    for clst in CC:
        mean_cluster_size += len(clst)
    mean_cluster_size /= n_clusters
    for clst in CC:
        std_cluster_size += (len(clst)-mean_cluster_size)**2
    std_cluster_size /= n_clusters-1
    std_cluster_size = np.sqrt(std_cluster_size)

    ##############################

    # horrifying monstrosity: associate every vertex with perimeter
    # of the non-clustered neighbors
    # correct sigmas:
    sig_12 = np.sqrt((sigma_12**2).sum(axis=-1))/2
    sig_20 = np.sqrt((sigma_20**2).sum(axis=-1))/2
    sig_01 = np.sqrt((sigma_01**2).sum(axis=-1))/2
    vtx_perim = np.zeros(nvtx, dtype=np.float64)
    add_to_1_2 = labeled_vtx[tlist[:, 1]] != labeled_vtx[tlist[:, 2]]
    numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 1],
                     sig_12[add_to_1_2])
    numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 2],
                     sig_12[add_to_1_2])
    add_to_2_0 = labeled_vtx[tlist[:, 2]] != labeled_vtx[tlist[:, 0]]
    numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 2],
                     sig_20[add_to_2_0])
    numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 0],
                     sig_20[add_to_2_0])
    add_to_0_1 = labeled_vtx[tlist[:, 0]] != labeled_vtx[tlist[:, 1]]
    numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 0],
                     sig_01[add_to_0_1])
    numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 1],
                     sig_01[add_to_0_1])

    perim = vtx_perim[active].sum()

    return (
            total_volume, total_area,
            gy_eig[0], gy_eig[1], gy_eig[2],
            nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size
            )


@njit
def calculate_statistic_old_w(active, pos, blist, tlist, bending_E):
    """Calculate the statistics from the arrays."""
    nvtx = len(active)

    t_normals = np.zeros((tlist.shape[0], 3))

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    ######################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    ##############################################################
    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (active[blist[:, 0]] & active[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    ######################################################
    # mean curvature:
    # new version saves it, but we need sigmas for the perimeters
    # a lot harder, since we don't have the neighbors directly.
    # the components of summation are,
    # for each vertex i:
    #   sum all l_ij * cotan(theta_opposite)/2 --> rh[i]
    #   sum normal of triangles (to determine h sign) --> tnh[i]
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # add the normal to each vertex in the triangle:
    # vtx_normal[tri->vtx[0]] += tri->normal. then for 1 and 2
    # problematic due to repeated indices in triangles- two triangles can
    # have the same vertex in 0, screwing the +=
    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)
    # we only need direction, tnh*rh<0, so no need to normalize

    # Summing the other part is more difficult
    # we go on each vertex of the triangle k=[0,1,2]
    # calculate cotan(theta[k])
    # and add the relevant lij*cotan(theta[k])/2 vector to rh[i!=k]

    # To get cotan, we will beed bond length square
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    # numpy vectorized version of the c calculation
    # cot[q] = |a||b|cos/sqrt(|a|^2|b|^2 - |a|^2|b|^2cos^2)
    # |a||b|cos = a @ b
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    # contributions to 1 and 2: +-l_12 * cot(theta[0])=+-sigma12
    # (divide by 2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    # few! that was not nice

    #############################################################
    # perimeter: if vertex "i" in a triangle is unique, there's a
    # domain boundary running through: |sigma_ij|+|sigma_ik|
    # perim = 0
    # # for any 0th vertex that is unique, 0!=1 and 1==2
    # unique_vtx = ((active[tlist[:, 0]] != active[tlist[:, 1]])
    #               & (active[tlist[:, 1]] == active[tlist[:, 2]]))
    # # += |sigma_20|
    # perim += np.sqrt((sigma_20[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # += |sigma_01|
    # perim += np.sqrt((sigma_01[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # same for any 1th vertex that is unique, 1!=2 and 2==0
    # unique_vtx = ((active[tlist[:, 1]] != active[tlist[:, 2]])
    #               & (active[tlist[:, 2]] == active[tlist[:, 0]]))
    # perim += np.sqrt((sigma_01[unique_vtx, :]**2).sum(axis=-1)).sum()
    # perim += np.sqrt((sigma_12[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # same for any 2th vertex that is unique, 2!=0 and 0==1
    # unique_vtx = ((active[tlist[:, 2]] != active[tlist[:, 0]])
    #               & (active[tlist[:, 0]] == active[tlist[:, 1]]))
    # perim += np.sqrt((sigma_12[unique_vtx, :]**2).sum(axis=-1)).sum()
    # perim += np.sqrt((sigma_20[unique_vtx, :]**2).sum(axis=-1)).sum()
    # # sigmas are still 2*sigma
    # perim /= 2

    ####################################
    # cluster size distribution:

    CC, labeled_vtx = connected_components(nvtx, blist, active)
    n_clusters = len(CC)
    labeled_vtx[~active] = n_clusters
    mean_cluster_size = 0.
    std_cluster_size = 0.
    for clst in CC:
        mean_cluster_size += len(clst)
    mean_cluster_size /= n_clusters
    for clst in CC:
        std_cluster_size += (len(clst)-mean_cluster_size)**2
    std_cluster_size /= n_clusters-1
    std_cluster_size = np.sqrt(std_cluster_size)

    ##############################
    # get statistics of individual clusters
    # stats: size, perim, l, E
    clst_size = np.zeros(n_clusters, dtype=np.int64)
    clst_perim = np.zeros(n_clusters, dtype=np.float64)
    lam = np.zeros((n_clusters, 3), dtype=np.float64)
    clst_E = np.zeros(n_clusters, dtype=np.float64)
    clst_sample_id = np.zeros(n_clusters, dtype=np.int64)

    # horrifying monstrosity: associate every vertex with perimeter
    # of the non-clustered neighbors
    # correct sigmas:
    sig_12 = np.sqrt((sigma_12**2).sum(axis=-1))/2
    sig_20 = np.sqrt((sigma_20**2).sum(axis=-1))/2
    sig_01 = np.sqrt((sigma_01**2).sum(axis=-1))/2
    vtx_perim = np.zeros(nvtx, dtype=np.float64)
    add_to_1_2 = labeled_vtx[tlist[:, 1]] != labeled_vtx[tlist[:, 2]]
    numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 1],
                     sig_12[add_to_1_2])
    numpy_sum_extend(vtx_perim, tlist[add_to_1_2, 2],
                     sig_12[add_to_1_2])
    add_to_2_0 = labeled_vtx[tlist[:, 2]] != labeled_vtx[tlist[:, 0]]
    numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 2],
                     sig_20[add_to_2_0])
    numpy_sum_extend(vtx_perim, tlist[add_to_2_0, 0],
                     sig_20[add_to_2_0])
    add_to_0_1 = labeled_vtx[tlist[:, 0]] != labeled_vtx[tlist[:, 1]]
    numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 0],
                     sig_01[add_to_0_1])
    numpy_sum_extend(vtx_perim, tlist[add_to_0_1, 1],
                     sig_01[add_to_0_1])

    perim = vtx_perim[active].sum()
    for i, clst_list in enumerate(CC):
        clst = np.array(clst_list)
        clst_size[i] = len(clst)
        clst_E[i] = bending_E[clst].sum()
        clst_gy_eig = np.linalg.eigvalsh(
            (pos[clst].T @ pos[clst]) / clst_size[i])
        clst_gy_eig.sort()
        lam[i, :] = clst_gy_eig
        clst_perim[i] = vtx_perim[clst].sum()
        clst_sample_id[i] = clst[0]

    return (
            (total_volume, total_area,
             gy_eig[0], gy_eig[1], gy_eig[2],
             nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size),
            (clst_size, clst_E, lam[:, 0], lam[:, 1], lam[:, 2], clst_perim,
             clst_sample_id)
            )


def get_statistics_from_vtu(vtu_path, v=False, w=True):
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
        curr_node = tree.find('.//*[@Name=\"curvature\"]')
        c = xml_to_np(curr_node)
        # curr_node = tree.find('.//*[@Name=\"normal\"]')
        # v_normals = xml_to_np(curr_node)
        curr_node = tree.find('.//*[@Name=\"spontaneous_curvature\"]')
        c0 = xml_to_np(curr_node)
        if (nodetype == nodetype[0]).any():
            # attempt to reconstruct different vertices
            if (c0 != c0[0]).any():
                nodetype[c0 != c0[0]] |= 2 | 256  # added "active"
            else:
                # try w?
                curr_node = tree.find('.//*[@Name=\"bonding_strength\"]')
                w_bond = xml_to_np(curr_node)
                if (w_bond != w_bond[0]):
                    nodetype[w_bond != w_bond[0]] |= 1 | 256

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

    if new_ver_vtu:
        if w == 0 or w is None:
            out = calculate_statistic_new_ww(nodetype, c, pos,
                                             blist, tlist, bending_E)
            return new_ver_vtu, out
        if w == 1:
            out, df = calculate_statistic_new_ww(nodetype, c, pos,
                                                 blist, tlist, bending_E)
            return new_ver_vtu, out, df
        if w == 2:
            out, df1, df2 = calculate_statistic_new_ww(nodetype, c, pos,
                                                       blist, tlist, bending_E)
            return new_ver_vtu, out, df1, df2
    else:
        if w:
            outs, df = calculate_statistic_old_w(active, pos, blist, tlist,
                                                 bending_E)
            return new_ver_vtu, outs, df
        else:
            outs = calculate_statistic_old(active, pos, blist, tlist)
            return new_ver_vtu, outs


def main():
    """Parse command line args to find .vtu files, generates statistics.

    Takes any number of vtu files or directory, consolidate statistics
    to a single file (in alphabetical order) specified by
    -o filename.csv (default pystatistics.csv). Statistics are:
    No, Volume, Area, lamdba1, lambda2, lambda3, Nbw/Nb,
    hbar, mean_cluster_size, std_cluster_size, and line_length
    more options:
    -v: verbosity, yap endlessly
    -w: write a histogram for each vtu

    for example:
    >$python statistics_from_vtu.py timestep_000*
    Creates pystatistics.csv:
        0, volume, area, ... linelength
        1, Volume, area, ... linelength
    >$python statistics_from_vtu.py . -o stat -w
    Creates stat.csv:
        0, volume, area, ... linelength
        1, Volume, area, ... linelength
    and historgams
    |-histogrma_000001.csv
    |-histogrma_000002.csv
    |-histogrma_000014.csv
    (notice that it doesn't write the file number into No,
     it just goes by alphabetical order)
    """
    # parse the arguments:
    parser = argparse.ArgumentParser(
        description='''Get statistics of major order parameters from vtus
        files. statistics are sorted alphabetically
        ''')
    parser.add_argument("vtu_in", nargs='+',
                        help='.vtu files or directories with .vtu files')
    parser.add_argument("-o", "--out-file", help='.csv file name to write',
                        default='pystatistics.csv')
    parser.add_argument("-v", "--verbose", help='increase output verbosity',
                        action="store_true")
    parser.add_argument("-w", "--write-cluster", help='write histogram files',
                        action="count")
    args = parser.parse_args()

    new_file = args.out_file  # name of file to create
    # if just name: make here. must end with .csv
    new_file = os.path.splitext(new_file)[0] + '.csv'
    v = args.verbose  # ease verbosity checks
    w = args.write_cluster

    # make the files input iterable, even if there is only one file
    if not isinstance(args.vtu_in, list):
        args.vtu_in = [args.vtu_in]

    # get the files
    # could be mixed list of files and directories
    all_vtus = []
    for vtu_in in args.vtu_in:

        # seperate the case of one file vs. one directory
        if v:
            print('discerning input ', vtu_in, '\n')

        # for directory, get all .vtu files
        if os.path.isdir(vtu_in):
            if v:
                print('directory: extracting files')
            with os.scandir(vtu_in) as it:
                strings = [entry.path for entry in it]

        # from each file: get that file,
        elif os.path.isfile(vtu_in):
            if v:
                print('file: extracting')
            strings = [vtu_in]  # encase for same handling as directory case

        # if not a file or directory: error
        else:
            raise ValueError('input ', args.vtu_in, 'is not file or directory')

        # get only .vtu files
        vtus = [s for s in strings if s.endswith('.vtu')]
        if v:
            print("got ", len(vtus), " files", vtus[:5], "...")

        all_vtus.extend(vtus)

    # sort alphabetically:
    all_vtus.sort()

    stat_header = ["vtu", "volume", "area", "lambda1", "lambda2",
                   "lambda3", "bond_ratio", "mean_curvature", "line_length",
                   "mean_cluster_size", "std_cluster_size"]
                   #"cluster_size_per_vertex"]

    ##########################################
    # now has all vtu files. For each file:
    # Calculate and return statistics and
    # potentially write a histogram_*.csv

    # debug: regular, non multiprocessing:
    stat_futures = []
    for vtu in all_vtus:
        out = get_statistics_from_vtu(vtu, v, w)
        _w = w
        if not out[0] and _w == 2:
            # found old version: doesn't dupport w==2
            _w = 1
        if _w == 0 or _w is None:
            _, pystats = out
        elif _w == 1:
            _, pystats, df_data = out
            df_header = ['cluster_size', 'bending_energy',
                         'lambda1', 'lambda2', 'lambda3', 'perim', 'id']
            df = pd.DataFrame({key: value for
                               key, value in zip(df_header, df_data)})
            #df.to_csv(vtu.replace('timestep', 'histogram'))
            # write df
        elif _w == 2:
            _, pystats, df1_data, df2_data = out
            df1_header = ['cluster_size', 'bending_energy',
                          'lambda1', 'lambda2', 'lambda3', 'perim', 'id']
            df1 = pd.DataFrame({key: value for
                                key, value in zip(df1_header, df1_data)})
            df2_header = ['type', 'cluster_size', 'bending_energy',
                          'lambda1', 'lambda2', 'lambda3', 'perim', 'id']
            df2 = pd.DataFrame({key: value for
                                key, value in zip(df2_header, df2_data)})
            # write df
        stat_futures.append((vtu, *pystats))

    if v:
        print("writing to main statistics file ", new_file)

    # write main statistics file
    df = pd.DataFrame(stat_futures, columns=stat_header)
    #df.to_csv(new_file)


if __name__ == "__main__":
    main()
