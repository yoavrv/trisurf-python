#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for numeric functions accelerated by numba.

Split to small njitted functions to re-derive trisurf properties

FUNCTIONS:
    - numpy_sum_extend(A,B,C): equivalent to A[B]+=D
    - connected_components(nvtx, blist, v_keep): clustering algorithm

    - 5 permutation of calculate_statistic_[new/old][/_w/_ww]:
        get trisurf statistics from vtu-derived numpy array as numpy arrays

Created on Thu Aug 12 14:26:31 2021

@author: yoav
"""

from functools import wraps
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(f=None, *args, **kwargs):
        """Fake do-nothing njit since we have no numba"""

        def decorator(func):
            return func 

        if callable(f):
            return f
        else:
            return decorator

#%%
############################
# small njitted functions: #
############################
# extension of numpy sum
# and implementation of connected component that works with numba


@njit
def numpy_sum_extend(array_to_add_to, array_extend_indices, array_to_add_from):
    """Apply A[B] += D, even when B and D are larger than A."""
    for i, j in enumerate(array_extend_indices):
        array_to_add_to[j, ...] += array_to_add_from[i, ...]


@njit
def _strong_connect(i, c_idx, S, CC, v_idx, v_lowlink,
                    v_on_stack, blist, v_keep=None):
    """Recursive subfunction of connected_components."""
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
            _strong_connect(j, c_idx, S, CC,
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

    vtx are assumed to be their indices: vertex[i]==i.
    Mostly copied from Wikipedia: "Trajan's strongly connected-
    components algorithm"
    (which can't be too good: we're not having any strong connection)

    blist: mx2 array connecting vertices
    blist[m,:]==[i,j] is the mth edge connecting vertices i,j
    v_keep: boolean arrays size nvtx, saying if this is a relevant vertex
    irrelevent vertices are skipped

    returns list of list of nodes i.e. list of clusters, and a cluster_id array
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
            _strong_connect(i, c_idx, S, CC, v_idx, v_lowlink,
                            v_on_stack, blist, v_keep)
    for i, clst in enumerate(CC):
        for node in clst:
            v_lowlink[node] = i
    return CC, v_lowlink

#%%
####################################
# individual processing functions: #
####################################
# get various quantities from PyVtu arrays
# not bundled into the big functions

@njit(cache=True, error_model='numpy')
def bonding_ratio(blist, bonding):
    """Get bonding ratio from blist and bonding list."""

    # get bonds with energy
    # bonds_with_e = bonding[bond->vtx[0]] and bonding[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])
    return nbw_nb


@njit(cache=True, error_model='numpy')
def gyration_eigenvalues(pos):
    """Get gyration eigenvalues from position."""
    nvtx = pos.shape[0]
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is equivalent to G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()
    return gy_eig


@njit(cache=True, error_model='numpy')
def bounding_radius(pos):
    """Get bounding radius from position."""
    nvtx = pos.shape[0]
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    return np.sqrt(max((pos**2).sum(axis=1)))


@njit(cache=True, error_model='numpy')
def area_volume(pos, tlist):
    """Get area and volume from position and tlist."""
    nvtx = pos.shape[0]

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

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

    return total_area, total_volume

@njit(cache=True, error_model='numpy')
def voroni_bonds(pos, tlist):
    """Calculate voronoi (dual) bonds from position and connectivity list"""
    nvtx = pos.shape[0]

    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # shorthand to simplify (n)*(mxn) array operations addim1(x[:n])*y[:n,:]
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    sigma = 0.0*xyz0
    area = sigma + 0.0

    # To get cotan, we will need bond length square
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
    sigma[:, 0] = addim1(cot_at) * (xyz2 - xyz1)
    area[:, 0] = (cot_at*bond_sqr12)/8

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma[:,1] = addim1(cot_at) * (xyz0 - xyz2)
    area[:,1] = (cot_at*bond_sqr02)/8
    # contributions to 2 and 0:

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma[:,2] = addim1(cot_at) * (xyz1 - xyz0)
    area[:,2] = (cot_at*bond_sqr01)/8
    # contributions to 1 and 2:
    
    return sigma, area

@njit(cache=True, error_model='numpy')
def curvature(pos, tlist):
    """Calculate curvature from position and connectivity list.

    Return curvature per vertex and area per vertex"""
    nvtx = pos.shape[0]

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # shorthand to simplify (n)*(mxn) array operations addim1(x[:n])*y[:n,:]
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    areas = np.sqrt((t_normals**2).sum(axis=-1))/2

    # the components of summation are,
    # for each vertex i:
    #   sum all l_ij * cotan(theta_opposite)/2 --> rh[i]
    #   sum normal of triangles (to determine h sign) --> tnh[i]
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)
    s = np.zeros(nvtx)

    # summing the normals is easy, since we have them from volume/area,
    # but we didn't normalize them
    t_normals /= addim1(2*areas)  # normalizing vectors was skipped

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
    area_12 = (cot_at*bond_sqr12)/8
    # contributions to 1 and 2: +-l_12 * cot(theta[0])=+-sigma12
    # (divide by 2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)
    numpy_sum_extend(s, tlist[:, 1], area_12)
    numpy_sum_extend(s, tlist[:, 2], area_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    area_20 = (cot_at*bond_sqr02)/8
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)
    numpy_sum_extend(s, tlist[:, 2], area_20)
    numpy_sum_extend(s, tlist[:, 0], area_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    area_01 = (cot_at*bond_sqr01)/8
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)
    numpy_sum_extend(s, tlist[:, 0], area_01)
    numpy_sum_extend(s, tlist[:, 1], area_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    h /= s

    # few! that was not nice
    return h, s


@njit(cache=True, error_model='numpy')
def perimeter(vtype, pos, blist, tlist):
    """Get perimeter based on nodetype"""
    nvtx = pos.shape[0]
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # shorthand to simplify (n)*(mxn) array operations addim1(x[:n])*y[:n,:]
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    # position of each point in triangle
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]

    ######################################################
    # perimeter:
    # we need the dual bond in each triangle

    # To get cotan, we will beed bond length square
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    # dual bond
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)

    #######################
    # typed clusters:
    type_blist = blist[vtype[blist[:, 0]] == vtype[blist[:, 1]], :]
    CC2, labeled_vtx = connected_components(nvtx, type_blist)

    ############################################################
    # horrifying monstrosity: associate every vertex with perimeter
    # of the non-clustered neighbors.
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
    perim = vtx_perim.sum()/2  # every boundary is shared between two vertices

    return perim


@njit(cache=True, error_model='numpy')
def segregation_factor(vtype, blist, ignore_type=4):
    """Calculate segregation factor.

    p_same is the fraction of bonds between vertices of the same type, ignoring
    connections to ignore_type. segregation factor is 2*p_same-1:
    [0,1) represent mixed and seperated types
    [-1,0) represent anti-correlated ("chekerboard") arrangement.
    """
    btype = blist.copy()
    btype[:, 0] = vtype[blist[:, 0]]
    btype[:, 1] = vtype[blist[:, 1]]
    b = btype[((btype[:, 0] != ignore_type) & (btype[:, 1] != ignore_type)), :]
    p_same = (b[:, 0] == b[:, 1]).mean()
    return 2*p_same-1


def segregation_factor2(vtype, blist, keep_mask=None, placeholder=999):
    """Calculate segregation factor for vertices keep_mask[v]==True.
    
    A utility function for segregation_factor
    """
    if keep_mask is None:
        return segregation_factor(vtype, blist)
    else:
        vtype2 = vtype.copy()
        vtype2[~keep_mask] = placeholder
        return segregation_factor(vtype2, blist, ignore_type=placeholder)


@njit(cache=True, error_model='numpy')
def nematic_order(vtype, blist, director, is_anisotropic_vtx=8):
    """Calculate nematic order
    
    Adds and averages CMC neighbors over their orientation
    S = (3 d1@d2 - 1)/2
    """
    nematic_order = np.zeros(vtype.shape, dtype=np.float64)
    n = np.zeros(vtype.shape, dtype=np.int64)
    for i,j in blist:
        if vtype[i]&is_anisotropic_vtx and vtype[j]&is_anisotropic_vtx:
            l = director[i,:]@director[j,:]
            nematic_order[i]+=0.5*(3*(l**2)-1)
            nematic_order[j]+=0.5*(3*(l**2)-1)
            n[i]+=1
            n[j]+=1
    n[n==0]=1 # prevent division by 0
    return nematic_order/n


# for i in range(len(S2)):
#     if v.type[i]&8:
#         neis = v.get_neighbors(i)
#         vnei = neis[(v.type[neis]&8) == 8]
#         if len(vnei):
#             print('here',i,vnei)
#             S2[i] = (0.5*( 3* (v.director[vnei]@v.director[i])**2 - 1)).mean()



#%%
###################################################
# Large numeric functions for getting statistics: #
###################################################
# njitted calculate_statistics [new, old] x [_ w ww], nonjitted get_statistics
# njitted functions: take numpy arrays extracted from vtu and return
# various statistics
# incredibly repetetive: five function due to
# slightly different argument and return types
# but they are all broadly the same


@njit(cache=True, error_model='numpy')
def calculate_statistic_new_ww(node_type, pos, blist, tlist,
                               bending_E, force):
    """Get statistics and typed cluster info from arrays of a new-typed .vtu.

    Takes node_type[nvtx], position[nvtx,3], blist[nbonds,2], tlist[ntri, 3],
    and bending_energy[nvtx], extract statistics:

    returns (out, df1, df2) where
    out: main statistics     (volume, area, lambda1,2,3, bonding ratio,
                              mean curvature, perimeter,
                              mean cluster size, std cluster size,
                              force per vertex)

    df1: clusters data       (size, energy, lambda1 2,3, perimeter,
                              total force x,y,z,
                             sample id of one of the vertices)

    df2: typed cluster data  (type, size, energy, lambda1,2,3, perimeter,
                              total force x,y,z,
                              sample id of one of the vertices)

    main statistics is meant to be concatenated to a cumulative dataframe
    The dfs are meant to be constructed into individual pandas dataframes
    (sample id: make it possible to find each cluster in the simulation later)
    """
    nvtx = len(node_type)
    bonding = node_type & 1 != 0
    active = node_type & 2 != 0

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # shorthand to simplify (n)*(mxn) array operations addim1(x[:n])*y[:n,:]
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
    # get bounding sphere radius
    bounding_R = np.sqrt(max((pos**2).sum(axis=1)))

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
    # get force per vertex
    nactive = active.sum()
    if nactive == 0:
        force_mag = 0
    else:
        force_mag = np.sqrt((force[active]**2).sum(axis=1)).sum()
        force_mag /= nactive

    ####################################
    # cluster size distribution:

    if not bonding.any() and active.any():
        CC, labeled_vtx = connected_components(nvtx, blist, active)
        n_clusters = len(CC)
        labeled_vtx[~active] = n_clusters
    else:
        CC, labeled_vtx = connected_components(nvtx, blist, bonding)
        n_clusters = len(CC)
        labeled_vtx[~bonding] = n_clusters

    if n_clusters == 0:
        mean_cluster_size = 0
        mean_cluster_size_per_vertex = 0
        std_cluster_size = np.nan
        std_cluster_size_per_vertex = np.nan
        perim = 0
    else:
        mean_cluster_size = 0.
        mean_cluster_size_per_vertex = 0
        std_cluster_size = 0.
        std_cluster_size_per_vertex = 0.
        clustered = 0
        for clst in CC:
            mean_cluster_size_per_vertex += len(clst)*len(clst)
            mean_cluster_size += len(clst)
            clustered += len(clst)
        mean_cluster_size_per_vertex /= clustered
        mean_cluster_size /= n_clusters

        for clst in CC:
            std_cluster_size_per_vertex += ((len(clst)
                                             - mean_cluster_size_per_vertex)**2
                                            ) * len(clst)
            std_cluster_size += ((len(clst) - mean_cluster_size)**2)
        std_cluster_size_per_vertex /= clustered-1
        std_cluster_size_per_vertex = np.sqrt(std_cluster_size_per_vertex)
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
    clst_force = np.zeros((n_clusters, 3), dtype=np.float64)

    for i, clst_list in enumerate(CC):
        clst = np.array(clst_list)
        clst_size[i] = len(clst)
        clst_E[i] = bending_E[clst].sum()
        clst_pos = pos[clst]
        clst_pos = clst_pos - clst_pos.sum(axis=0)/clst_size[i]
        clst_gy_eig = np.linalg.eigvalsh(
            (clst_pos.T @ clst_pos) / clst_size[i])
        clst_gy_eig.sort()
        lam[i, :] = clst_gy_eig
        clst_perim[i] = vtx_perim[clst].sum()
        clst_sample_id[i] = clst.min()+nvtx*clst.max()
        # getitem doesn't work
        for v in clst_list:
            clst_force[i, :] += force[v, :]

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
    clst_force_typed = np.zeros((n_clusters, 3), dtype=np.float64)
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
        clst_pos = pos[clst]
        clst_pos = clst_pos - clst_pos.sum(axis=0)/clst_size_typed[i]
        clst_gy_eig = np.linalg.eigvalsh(
            (clst_pos.T @ clst_pos) / clst_size_typed[i])
        clst_gy_eig.sort()
        lam_typed[i, :] = clst_gy_eig
        clst_perim_typed[i] = vtx_perim[clst].sum()
        # getitem doesn't work
        for v in clst_list:
            clst_force_typed[i, :] += force[v, :]
        clst_sample_id_typed[i] = clst.min()+nvtx*clst.max()
        clst_type[i] = node_type[clst[0]]

    return (
            (total_volume, total_area, bounding_R,
             gy_eig[0], gy_eig[1], gy_eig[2],
             nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size,
             force_mag,
             mean_cluster_size_per_vertex, std_cluster_size_per_vertex),
            (clst_size, clst_E, lam[:, 0], lam[:, 1], lam[:, 2], clst_perim,
             clst_force[:, 0], clst_force[:, 1], clst_force[:, 2],
             clst_sample_id),
            (clst_type, clst_size_typed, clst_E_typed,
             lam_typed[:, 0], lam_typed[:, 1], lam_typed[:, 2],
             clst_perim_typed, clst_force_typed[:, 0], clst_force_typed[:, 1],
             clst_force_typed[:, 2], clst_sample_id_typed)
            )


@njit(cache=True, error_model='numpy')
def calculate_statistic_new_w(node_type, pos, blist, tlist,
                              bending_E, force):
    """Get statistics and cluster info from arrays of a new-typed .vtu.

    Takes node_type[nvtx], position[nvtx,3], blist[nbonds,2], tlist[ntri, 3],
    and bending_energy[nvtx], extract statistics:

    returns (out, df) where
    out: main statistics     (volume, area, lambda1,2,3, bonding ratio,
                              mean curvaturem, perimeter,
                              mean cluster size, std cluster size,
                              force per vertex)

    df: clusters data        (size, energy, lambda1 2,3, perimeter,
                              total force x, y, z,
                             sample id of one of the vertices)

    main statistics is meant to be concatenated to a cumulative dataframe
    The df is meant to be constructed into individual pandas dataframe
    (sample id: make it possible to find each cluster in the simulation later)
    """
    nvtx = len(node_type)
    bonding = node_type & 1 != 0
    active = node_type & 2 != 0

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    #################################################
    # get volume and area:

    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area:
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume:
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get bounding sphere radius
    bounding_R = np.sqrt(max((pos**2).sum(axis=1)))

    #################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)

    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    #################################################
    # get bonds with energy

    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    #################################################
    # mean curvature:

    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # tnh: add the normal to each vertex in the triangle:

    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)

    # rh:

    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    ####################################
    # get force per vertex
    nactive = active.sum()
    if nactive == 0:
        force_mag = 0
    else:
        force_mag = np.sqrt((force[active]**2).sum(axis=1)).sum()
        force_mag /= nactive

    #################################################
    # cluster size distribution:

    if not bonding.any() and active.any():
        CC, labeled_vtx = connected_components(nvtx, blist, active)
        n_clusters = len(CC)
        labeled_vtx[~active] = n_clusters
    else:
        CC, labeled_vtx = connected_components(nvtx, blist, bonding)
        n_clusters = len(CC)
        labeled_vtx[~bonding] = n_clusters

    if n_clusters == 0:
        mean_cluster_size = 0
        mean_cluster_size_per_vertex = 0
        std_cluster_size = np.nan
        std_cluster_size_per_vertex = np.nan
        perim = 0
    else:
        mean_cluster_size = 0.
        mean_cluster_size_per_vertex = 0
        std_cluster_size = 0.
        std_cluster_size_per_vertex = 0.
        clustered = 0
        for clst in CC:
            mean_cluster_size_per_vertex += len(clst)*len(clst)
            mean_cluster_size += len(clst)
            clustered += len(clst)
        mean_cluster_size_per_vertex /= clustered
        mean_cluster_size /= n_clusters

        for clst in CC:
            std_cluster_size_per_vertex += ((len(clst)
                                             - mean_cluster_size_per_vertex)**2
                                            ) * len(clst)
            std_cluster_size += ((len(clst) - mean_cluster_size)**2)
        std_cluster_size_per_vertex /= clustered-1
        std_cluster_size_per_vertex = np.sqrt(std_cluster_size_per_vertex)
        std_cluster_size /= n_clusters-1
        std_cluster_size = np.sqrt(std_cluster_size)

        # associate every vertex with perimeter

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

    #################################################
    # get statistics of individual clusters
    # stats: size, perim, l, E
    clst_size = np.zeros(n_clusters, dtype=np.int64)
    clst_perim = np.zeros(n_clusters, dtype=np.float64)
    lam = np.zeros((n_clusters, 3), dtype=np.float64)
    clst_E = np.zeros(n_clusters, dtype=np.float64)
    clst_force = np.zeros((n_clusters, 3), dtype=np.float64)
    clst_sample_id = np.zeros(n_clusters, dtype=np.int64)

    for i, clst_list in enumerate(CC):
        clst = np.array(clst_list)
        clst_size[i] = len(clst)
        clst_E[i] = bending_E[clst].sum()
        clst_pos = pos[clst]
        clst_pos = clst_pos - clst_pos.sum(axis=0)/clst_size[i]
        clst_gy_eig = np.linalg.eigvalsh(
            (clst_pos.T @ clst_pos) / clst_size[i])
        clst_gy_eig.sort()
        lam[i, :] = clst_gy_eig
        clst_perim[i] = vtx_perim[clst].sum()
        for v in clst_list:
            clst_force[i, :] += force[v, :]
        clst_sample_id[i] = clst.min()+nvtx*clst.max()

    return (
            (total_volume, total_area, bounding_R,
             gy_eig[0], gy_eig[1], gy_eig[2],
             nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size,
             force_mag,
             mean_cluster_size_per_vertex, std_cluster_size_per_vertex,),
            (clst_size, clst_E, lam[:, 0], lam[:, 1], lam[:, 2], clst_perim,
             clst_force[:, 0], clst_force[:, 1], clst_force[:, 2],
             clst_sample_id)
            )


@njit(cache=True, error_model='numpy')
def calculate_statistic_new(node_type, pos, blist, tlist,
                            bending_E, force):
    """Get statistics from arrays of a new-typed .vtu.

    Takes node_type[nvtx], position[nvtx,3], blist[nbonds,2], tlist[ntri, 3],
    and bending_energy[nvtx], extract statistics:

    returns (main_statistics):
    (volume, area, lambda1,2,3, bonding ratio, mean curvaturem, perimeter,
     mean cluster size, std cluster size)

    main statistics is meant to be concatenated to a cumulative dataframe
    i.e. pd.Dataframe([main_statistics0, main_statistics1, ...],
                      cols=['volume', 'area'... 'std_cluster_size'])
    """
    nvtx = len(node_type)
    bonding = node_type & 1 != 0
    active = node_type & 2 != 0

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

    # helper function (macro-like)
    def addim1(x): return np.expand_dims(x, 1)  # inline plz

    #################################################
    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    t_normals = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area:
    double_areas = np.sqrt((t_normals**2).sum(axis=-1))
    # volume:
    total_area = double_areas.sum()/2
    total_volume = -((xyz0 + xyz1 + xyz2)*t_normals).sum()/18

    ##################################################
    # get bounding sphere radius
    bounding_R = np.sqrt(max((pos**2).sum(axis=1)))

    #################################################
    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)

    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    #################################################
    # get bonds with energy

    nbw_nb = (bonding[blist[:, 0]] & bonding[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    #################################################
    # mean curvature:

    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # add the normal to each vertex in the triangle:

    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)

    # rh:

    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:

    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    ####################################
    # get force per vertex
    nactive = active.sum()
    if nactive == 0:
        force_mag = 0
    else:
        force_mag = np.sqrt((force[active]**2).sum(axis=1)).sum()
        force_mag /= nactive

    #################################################
    # cluster size distribution:
    if not bonding.any() and active.any():
        CC, labeled_vtx = connected_components(nvtx, blist, active)
        n_clusters = len(CC)
        labeled_vtx[~active] = n_clusters
    else:
        CC, labeled_vtx = connected_components(nvtx, blist, bonding)
        n_clusters = len(CC)
        labeled_vtx[~bonding] = n_clusters

    if n_clusters == 0:
        mean_cluster_size = 0
        mean_cluster_size_per_vertex = 0
        std_cluster_size = np.nan
        std_cluster_size_per_vertex = np.nan
        perim = 0
    else:
        mean_cluster_size = 0.
        mean_cluster_size_per_vertex = 0
        std_cluster_size = 0.
        std_cluster_size_per_vertex = 0.
        clustered = 0
        for clst in CC:
            mean_cluster_size_per_vertex += len(clst)*len(clst)
            mean_cluster_size += len(clst)
            clustered += len(clst)
        mean_cluster_size_per_vertex /= clustered
        mean_cluster_size /= n_clusters

        for clst in CC:
            std_cluster_size_per_vertex += ((len(clst)
                                             - mean_cluster_size_per_vertex)**2
                                            ) * len(clst)
            std_cluster_size += ((len(clst) - mean_cluster_size)**2)
        std_cluster_size_per_vertex /= clustered-1
        std_cluster_size_per_vertex = np.sqrt(std_cluster_size_per_vertex)
        std_cluster_size /= n_clusters-1
        std_cluster_size = np.sqrt(std_cluster_size)

        # associate every vertex with perimeter

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

    return (total_volume, total_area, bounding_R,
            gy_eig[0], gy_eig[1], gy_eig[2],
            nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size,
            force_mag,
            mean_cluster_size_per_vertex, std_cluster_size_per_vertex,
            )


@njit(cache=True, error_model='numpy')
def calculate_statistic_old_w(active, pos, blist, tlist, bending_E):
    """Get statistics and cluster info from arrays of a old-type .vtu.

    Takes active[nvtx], position[nvtx,3], blist[nbonds,2], tlist[ntri, 3],
    and bending_energy[nvtx], extract statistics:

    returns (out, df) where
    out: main statistics     (volume, area, lambda1,2,3, bonding ratio,
                              mean curvaturem, perimeter,
                              mean cluster size, std cluster size)

    df: clusters data        (size, energy, lambda1 2,3, perimeter,
                             sample id of one of the vertices)

    main statistics is meant to be concatenated to a cumulative dataframe
    The df is meant to be constructed into individual pandas dataframe
    (sample id: make it possible to find each cluster in the simulation later)
    """
    nvtx = len(active)

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

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
    # get bounding sphere radius
    bounding_R = np.sqrt(max((pos**2).sum(axis=1)))

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
    if n_clusters == 0:
        mean_cluster_size = 0
        mean_cluster_size_per_vertex = 0
        std_cluster_size = np.nan
        std_cluster_size_per_vertex = np.nan
        perim = 0
    else:
        mean_cluster_size = 0.
        mean_cluster_size_per_vertex = 0
        std_cluster_size = 0.
        std_cluster_size_per_vertex = 0.
        clustered = 0
        for clst in CC:
            mean_cluster_size_per_vertex += len(clst)*len(clst)
            mean_cluster_size += len(clst)
            clustered += len(clst)
        mean_cluster_size_per_vertex /= clustered
        mean_cluster_size /= n_clusters

        for clst in CC:
            std_cluster_size_per_vertex += ((len(clst)
                                             - mean_cluster_size_per_vertex)**2
                                            ) * len(clst)
            std_cluster_size += ((len(clst) - mean_cluster_size)**2)
        std_cluster_size_per_vertex /= clustered-1
        std_cluster_size_per_vertex = np.sqrt(std_cluster_size)
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
        clst_pos = pos[clst]
        clst_pos = clst_pos - clst_pos.sum(axis=0)/clst_size[i]
        clst_gy_eig = np.linalg.eigvalsh(
            (clst_pos.T @ clst_pos) / clst_size[i])
        clst_gy_eig.sort()
        lam[i, :] = clst_gy_eig
        clst_perim[i] = vtx_perim[clst].sum()
        clst_sample_id[i] = clst.min()+nvtx*clst.max()

    return (
            (total_volume, total_area, bounding_R,
             gy_eig[0], gy_eig[1], gy_eig[2],
             nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size,
             mean_cluster_size_per_vertex, std_cluster_size_per_vertex),
            (clst_size, clst_E, lam[:, 0], lam[:, 1], lam[:, 2], clst_perim,
             clst_sample_id)
            )


@njit(cache=True, error_model='numpy')
def calculate_statistic_old(active, pos, blist, tlist):
    """Get statistics and cluster info from arrays of a old-type .vtu.

    Takes active[nvtx], position[nvtx,3], blist[nbonds,2], tlist[ntri, 3],
    and bending_energy[nvtx], extract statistics:

    returns (main_statistics):
    (volume, area, lambda1,2,3, bonding ratio, mean curvaturem, perimeter,
     mean cluster size, std cluster size)

    main statistics is meant to be concatenated to a cumulative dataframe
    i.e. pd.Dataframe([main_statistics0, main_statistics1, ...],
                      cols=['volume', 'area'... 'std_cluster_size'])
    """
    nvtx = len(active)

    t_normals = np.zeros((tlist.shape[0], 3))
    CM_pos = pos.sum(axis=0)/nvtx  # remove Center Mass
    pos = pos-CM_pos

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
    # get bounding sphere radius
    bounding_R = np.sqrt(max((pos**2).sum(axis=1)))

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

    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    t_normals /= addim1(double_areas)  # normalizing vectors was skipped

    # tnh:
    numpy_sum_extend(tnh, tlist[:, 0], t_normals)
    numpy_sum_extend(tnh, tlist[:, 1], t_normals)
    numpy_sum_extend(tnh, tlist[:, 2], t_normals)

    # rh:
    bond_sqr01 = ((xyz1-xyz0)**2).sum(axis=1)
    bond_sqr02 = ((xyz2-xyz0)**2).sum(axis=1)
    bond_sqr12 = ((xyz2-xyz1)**2).sum(axis=1)

    # on 0th vtx of each triangle:
    dot_prod_at = ((xyz1-xyz0)*(xyz2-xyz0)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01*bond_sqr02 - dot_prod_at**2)
    sigma_12 = addim1(cot_at) * (xyz2 - xyz1)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = ((xyz2-xyz1)*(xyz0-xyz1)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr01 - dot_prod_at**2)
    sigma_20 = addim1(cot_at) * (xyz0 - xyz2)
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = ((xyz0-xyz2)*(xyz1-xyz2)).sum(axis=-1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12*bond_sqr02 - dot_prod_at**2)
    sigma_01 = addim1(cot_at) * (xyz1 - xyz0)
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # h per vertex, do the division by 2 we didn't do before
    h = np.sqrt((rh**2).sum(axis=-1))/2
    # -h if pointing the other way (maybe triangle vertex order: maybe -?)
    h[(rh*tnh).sum(axis=-1) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    ####################################
    # cluster size distribution:

    CC, labeled_vtx = connected_components(nvtx, blist, active)
    n_clusters = len(CC)
    labeled_vtx[~active] = n_clusters
    if n_clusters == 0:
        mean_cluster_size = 0
        mean_cluster_size_per_vertex = 0
        std_cluster_size = np.nan
        std_cluster_size_per_vertex = np.nan
        perim = 0
    else:
        mean_cluster_size = 0.
        mean_cluster_size_per_vertex = 0
        std_cluster_size = 0.
        std_cluster_size_per_vertex = 0.
        clustered = 0
        for clst in CC:
            mean_cluster_size_per_vertex += len(clst)*len(clst)
            mean_cluster_size += len(clst)
            clustered += len(clst)
        mean_cluster_size_per_vertex /= clustered
        mean_cluster_size /= n_clusters

        for clst in CC:
            std_cluster_size_per_vertex += ((len(clst)
                                             - mean_cluster_size_per_vertex)**2
                                            ) * len(clst)
            std_cluster_size += ((len(clst) - mean_cluster_size)**2)
        std_cluster_size_per_vertex /= clustered-1
        std_cluster_size_per_vertex = np.sqrt(std_cluster_size)
        std_cluster_size /= n_clusters-1
        std_cluster_size = np.sqrt(std_cluster_size)

    ##############################

    # associate every vertex with perimeter

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
            total_volume, total_area, bounding_R,
            gy_eig[0], gy_eig[1], gy_eig[2],
            nbw_nb, hmean, perim, mean_cluster_size, std_cluster_size,
            mean_cluster_size_per_vertex, std_cluster_size_per_vertex,
            )
