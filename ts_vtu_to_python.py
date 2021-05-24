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
import os
import csv


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

# functions for the outside


def vtu_get_geometry(vtu_location):
    """Take path '/here/file.vtu', returns vertices, bonds, and triangles.

    Takes string, returns tuple:
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

    Takes string, returns tuple:
    spontaneus curvature [0, 0, 0.5, 0, 0.5,...],
    bending energy [6.143e-03, 1.577, ...]
    There's no point in taking the vertex id, which are just range(num_vertex)
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


def write_cluster_hist(clusters, vtu_location, v):
    """For statistics_from_vtu, write a histogram file for each vtu.

    Takes clusters, an array, vtu_location, the path of the vtu file,
    and v, the verbosity flag
    Writes an appropriate histogram file
    If vtu_location is "/over/here/blah.vtu"
    creates "/over/here/histogram_blah.csv"
    Special case: of name contains "timestep". replace
    i.e. if vtu_location is "/over/here/timestep_01.vtu"
    creates "/over/here/histogram_01.csv"
    """
    # create path for the histogram
    # replace suffix
    hist = vtu_location.replace('.vtu', '.csv')
    # affix histogram to the start of the name
    # if the name contains 'timestep'
    # e.g. it's in the 'timestep_000999.vtu' format
    # replace the 'timestep' instead
    base, filename = os.path.split(hist)
    if filename.__contains__('timestep'):
        filename = filename.replace('timestep', 'histogram')
    else:
        # add "histogram_" to the name
        filename = "histogram_" + filename
    hist = os.path.join(base, filename)

    if v:
        print('writing ', clusters[clusters != 0][:5], '... to ', hist)

    # write .csv file
    with open(hist, 'w+', newline='') as hist_file:
        writer = csv.writer(hist_file)
        writer.writerow(['cluster_size', 'number_of_clusters'])
        # write only rows with nonzero clusters
        relevant_rows = (x for x in enumerate(clusters, start=1)
                         if x[1] != 0)
        writer.writerows(relevant_rows)
