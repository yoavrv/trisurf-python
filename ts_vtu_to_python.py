#!/usr/bin/env python
# coding: utf-8

"""Collection of useful functions to parse a .vtu file into python.

This module, my first ever python module, provides
methods to parse a .vtu file, which is just an xml, and return
pythonically useful quantities, independant on the original
version of the trisurf
This is not quite trivial, as the .vtu also contains values that
are only relevant to the original c program

made by Yoav Ravid
Yoav.Ravid@weizmann.ac.il
"""

# imports
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse as sp


# useful internal functions

def _xml_to_str_list(node):
    r"""Take xml node, extract text array to list.

    Convert sequence to 1d array and lines to 2d arrays, e.g
     2 3\n4 5 6\n' -> [['2','3'],['4','5','6']]
    '\n 2 3 5' -> ['2','3','5']
    """
    all_text = node.text.strip()

    # if there are tabs in the middle, assume this is a 2d list
    if all_text.__contains__('\n'):
        return [x.split() for x in all_text.split('\n')]
    else:
        return all_text.split()

# functions  for the outside


def vtu_get_geometry(vtu_location):
    """Take path '/here/file.vtu', returns vertices, bonds, and triangles.

    Takes string, returns
    vertex positions, [[x1, y1, z1],[x2, y2, z2]...],
    bonds [[0,1],[0,5]...],
    triangles [[0,1,6],[0,2,8]...]
    """
    # parse vtu as xml file
    tree = ET.parse(vtu_location)
    root = tree.getroot()

    # get the geometry of the vesicle, contained in the first
    # (and only) piece of the Unstructured Grid
    ugrid = root.findall('UnstructuredGrid')
    vesicle = ugrid[0][0]  # "Yer a vesicle, Harry!"

    str_vtx_positions = _xml_to_str_list(
                            vesicle.findall('Points/DataArray')[0])
    vertex_positions = [[float(y) for y in x] for x in str_vtx_positions]

    str_cells = _xml_to_str_list(vesicle.findall('Cells/DataArray')[0])
    all_cells = [[int(y) for y in x] for x in str_cells]
    bonds = [x for x in all_cells if len(x) == 2]
    triangles = [x for x in all_cells if len(x) == 3]

    return vertex_positions, bonds, triangles


def vtu_get_tape(vtu_location):
    """Take path as string '/here/file.vtu',returns tape file.

    Takes string, returns string
    """
    # parse vtu as xml file
    tree = ET.parse(vtu_location)
    root = tree.getroot()

    # directly extract the one tape
    tape = root.findall('tape')[0]

    return tape.text

def vtu_get_vertex_data(vtu_location):
    """Take path '/here/file.vtu',returns curvature, bending energy.

    Takes string, returns
    spontaneus curvature [0, 0, 0.5, 0, 0.5,...],
    bending energy [6.143e-03, 1.577, ...]
    No point in taking the vertex id, which are just range(num_vertex)
    other energies are not always recorded
    """
    # parse vtu as xml file
    tree = ET.parse(vtu_location)
    root = tree.getroot()

    # extract additional vertex data
    c_txt = _xml_to_str_list(
        root.findall('.//*[@Name="spontaneous_curvature"]')[0])
    c = [float(x) for x in c_txt]

    b_txt = _xml_to_str_list(root.findall('.//*[@Name="bending_energy"]')[0])
    benergy = [float(x) for x in b_txt]

    return np.array(c), np.array(benergy)


def adjacency_from_bonds(bonds, keep=None, n_vertex=-1):
    """Take bond list, return sparse adjacency matrix.

    Takes bond, list of vertex pairs [[0, 1],[0, 3],[2, 7],...[101, 201]]
    May also take keep, list of vertex to keep (defaults to keep all)
    May also take n_vertex, so n_vertex^2 is size of the adjacency matrix
    If not given number of vertices, automatically calculates it
    """
    if n_vertex < 0:
        n_vertex = max(max(bonds))
    adj_mat = sp.lil_matrix((n_vertex, n_vertex), dtype=bool)
    if keep is None:
        for a, b in bonds:
            adj_mat[a, b] = True
            adj_mat[b, a] = True
    else:
        for a, b in bonds:
            if a in keep:
                if b in keep:
                    adj_mat[a, b] = True
                    adj_mat[b, a] = True
    return adj_mat


def cluster_dist_from_adjacency(adj, keep=None):
    """Take adjacency matrix and give cluster size distribution.

    Return a vector [n_1,n_2,...n_50], number of clusters of size 1,2,...50
    i.e. it zips with range(1, len(vec) + 1)
    """
    _, labeled_vtx = sp.csgraph.connected_components(adj)
    # first bincount converts the connected components to size of each cluster
    if keep is None:  # default behavior
        clusters_size = np.bincount(labeled_vtx)
    else:
        clusters_size = np.bincount(labeled_vtx[keep])
    # second bincount gets the cluster size distribution
    dist_size = np.bincount(clusters_size)
    return dist_size[1:]  # remove 0-sized cluster


def cluster_dist_from_bonds(bonds, keep=None, n_vertex=-1):
    """Take bond list, return cluster distribution.

    Direct of the adjacency_from_bonds function
    and cluster_dist_from_adjacency function
    """
    if n_vertex < 0:
        n_vertex = max(max(bonds))
    adj_mat = sp.lil_matrix((n_vertex, n_vertex), dtype=bool)
    if keep is None:
        for a, b in bonds:
            adj_mat[a, b] = True
            adj_mat[b, a] = True
    else:
        for a, b in bonds:
            if a in keep:
                if b in keep:
                    adj_mat[a, b] = True
                    adj_mat[b, a] = True
    _, labeled_vtx = sp.csgraph.connected_components(adj_mat)
    # first bincount converts the connected components to size of each cluster
    if keep is None:  # default behavior
        clusters_size = np.bincount(labeled_vtx)
    else:
        clusters_size = np.bincount(labeled_vtx[keep])
    # second bincount gets the cluster size distribution
    dist_size = np.bincount(clusters_size)
    return dist_size[1:]  # remove 0-sized cluster


def cluster_dist_from_vtu(vtu_location):
    """Take vtu location, return cluster distribution.

    Assuming node with spontaneous curvature c > 0 are active
    get cluster distribution of active vertices
    """
    # parse vtu as xml file
    tree = ET.parse(vtu_location)
    root = tree.getroot()

    # get the geometry of the vesicle, contained in the first
    # (and only) piece of the Unstructured Grid
    ugrid = root.findall('UnstructuredGrid')
    vesicle = ugrid[0][0]  # "Yer a vesicle, Harry!"

    str_cells = _xml_to_str_list(vesicle.findall('Cells/DataArray')[0])
    all_cells = [[int(y) for y in x] for x in str_cells]
    bonds = [x for x in all_cells if len(x) == 2]

    # extract additional vertex data
    c_txt = _xml_to_str_list(
        root.findall('.//*[@Name="spontaneous_curvature"]')[0])
    c = np.array([float(x) for x in c_txt])

    adj_mat = sp.lil_matrix((c.size, c.size), dtype=bool)
    for a, b in bonds:
        if c[a] > 0 and c[b] > 0:
            adj_mat[a, b] = True
            adj_mat[b, a] = True
    _, labeled_vtx = sp.csgraph.connected_components(adj_mat)
    # first bincount converts the connected components to size of each cluster
    clusters_size = np.bincount(labeled_vtx[c > 0])

    # second bincount gets the cluster size distribution
    dist_size = np.bincount(clusters_size)
    return dist_size[1:]  # remove 0-sized cluster
