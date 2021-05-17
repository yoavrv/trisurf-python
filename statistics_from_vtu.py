#!/usr/bin/env python
# coding: utf-8
"""Get statistics on a series of vtu files.

output as file
"""

# imports
import ts_vtu_to_python as v2p
import os
import concurrent.futures
import argparse
import csv
import numpy as np
import scipy.sparse as sp
from numba import njit


@njit  # supposednly, loop better on numpy than python
def numpy_sum_extend(array_to_add_to, array_extend_indices, array_to_add_from):
    """Apply A[B] += D, even when B and D are larger than A."""
    for i, j in enumerate(array_extend_indices):
        array_to_add_to[j, ...] += array_to_add_from[i, ...]


def get_statistics_from_vtu(vtu_location, v, w):
    """Get main statisticd (order parameters) from .vtu file.

    Takes vtu_location, the path of the vtu file, and v, the verbosity flag
    extract the geometry and calculates volume, area, gyration eigenvalues,
    active bond fraction, mean curvature, mean and std cluster size, and
    cmc-bare vesicle perimeter
    Optionally, takes -w and writes a histogram file
    and -v and being verbose
    """
    # Load geometry from file
    xyz, b, t = v2p.vtu_get_geometry(vtu_location)
    pos = np.array(xyz)
    blist = np.array(b)
    tlist = np.array(t)
    curvature, benergy = v2p.vtu_get_vertex_data(vtu_location)
    c = np.array(curvature)
    active = c > 0
    norms = np.zeros((tlist.shape[0], 3))
    nvtx = len(c)

    # get volume and area:
    xyz0 = pos[tlist[:, 0], :]
    xyz1 = pos[tlist[:, 1], :]
    xyz2 = pos[tlist[:, 2], :]
    norms = np.cross(xyz1 - xyz0, xyz2 - xyz0)
    # area = parallelogram/2 = |cross(AB,AC)|/2
    double_areas = np.linalg.norm(norms, axis=1)
    # volume: copy from c calculation
    # (triangle_area * norm * radius = signed area?)
    eighteen_volumes = np.einsum('ij,ij->i', (xyz0 + xyz1 + xyz2), norms)
    total_area = double_areas.sum()/2
    total_volume = -eighteen_volumes.sum()/18

    # get gyration eigenvalues G_mn = 1/N sum(r_n r_m)
    # which is G = (pos.T @ pos) / nvtx
    gy_eig = np.linalg.eigvalsh((pos.T @ pos) / nvtx)
    gy_eig.sort()

    # get bonds with energy
    # bonds_with_e = active[bond->vtx[0]] and active[bond->vtx[1]]
    # nbw = sum(bonds_with_e)
    nbw_nb = (active[blist[:, 0]] & active[blist[:, 1]]).sum()
    nbw_nb /= (blist.shape[0])

    # mean curvature:
    # a lot harder, since we don't have neighbors directly
    # the components of summation are
    # for each vertex i:
    # sum all l_ij * cotan(theta_opposite)/2
    # sum normal of triangles (to determine h sign)
    # this can be done on the triangle, which have well-determined neighbors
    rh = np.zeros(pos.shape)
    tnh = np.zeros(pos.shape)

    # summing the normals is now easy
    # (and we're looking for sign - no need to fuss about constants)
    norms /= double_areas[:, np.newaxis]  # normalize vectors, (was skipped)
    # add the normal to each vertex in the triangle:
    # vtx_normal[tri->vtx[0]] += tri->normal. then for 1 and 2
    # problematic due to repeated indices in triangles- two triangles can
    # have the same vertex in 0, screwing the +=
    numpy_sum_extend(tnh, tlist[:, 0], norms)
    numpy_sum_extend(tnh, tlist[:, 1], norms)
    numpy_sum_extend(tnh, tlist[:, 2], norms)

    # Summing other part is more difficult
    # we go on each vertex of the triangle
    # and add the relevant vector to rh
    # on 0th vtx of each triangle:
    bond_sqr01 = np.einsum('ij,ij->i', xyz1-xyz0, xyz1-xyz0)
    bond_sqr02 = np.einsum('ij,ij->i', xyz2-xyz0, xyz2-xyz0)
    bond_sqr12 = np.einsum('ij,ij->i', xyz2-xyz1, xyz2-xyz1)
    dot_prod_at = np.einsum('ij,ij->i', xyz1-xyz0, xyz2-xyz0)
    cot_at = dot_prod_at / np.sqrt(bond_sqr01 * bond_sqr02 - dot_prod_at ** 2)
    # 2*dual bond = 2*l_ij*(cot) (sigma is actually double_sigma)
    sigma_12 = cot_at[:, np.newaxis] * (xyz2 - xyz1)
    # contributions to 1 and 2: l_ij * cot (/2 later)
    numpy_sum_extend(rh, tlist[:, 1], sigma_12)
    numpy_sum_extend(rh, tlist[:, 2], -sigma_12)

    # on 1th vtx of each triangle
    dot_prod_at = np.einsum('ij,ij->i', xyz2 - xyz1, xyz0 - xyz1)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12 * bond_sqr01 - dot_prod_at ** 2)
    sigma_20 = cot_at[:, np.newaxis] * (xyz0 - xyz2)
    # contributions to 2 and 0:
    numpy_sum_extend(rh, tlist[:, 2], sigma_20)
    numpy_sum_extend(rh, tlist[:, 0], -sigma_20)

    # on 2th vtx
    dot_prod_at = np.einsum('ij,ij->i', xyz0 - xyz2, xyz1 - xyz2)
    cot_at = dot_prod_at / np.sqrt(bond_sqr12 * bond_sqr02 - dot_prod_at ** 2)
    sigma_01 = cot_at[:, np.newaxis] * (xyz1 - xyz0)
    # contributions to 1 and 2:
    numpy_sum_extend(rh, tlist[:, 0], sigma_01)
    numpy_sum_extend(rh, tlist[:, 1], -sigma_01)

    # total (taken from the c code), /2 we didn't do before
    h = np.sqrt(np.einsum('ij,ij->i', rh, rh))/2
    # -h if pointing the other way (maybe minus by vertex order)
    h[np.einsum('ij,ij->i', rh, tnh) < 0] *= -1
    hmean = h.sum() / (2 * total_area)

    # few! that was not nice

    # perimeter: each unique vertex "x" in a triangle has a boundary
    # betwween the others: |sigma_xy|+|sigma_xz|
    # sigmas are still twice
    perim = 0
    # 0th vertex is unique, 0!=1 and 1==2
    unique_vtx = ((active[tlist[:, 0]] != active[tlist[:, 1]])
                  & (active[tlist[:, 1]] == active[tlist[:, 2]]))
    perim += np.linalg.norm(sigma_20[unique_vtx, :], axis=1).sum()
    perim += np.linalg.norm(sigma_01[unique_vtx, :], axis=1).sum()
    # 1th vertex is unique, 1!=2 and 2==0
    unique_vtx = ((active[tlist[:, 1]] != active[tlist[:, 2]])
                  & (active[tlist[:, 2]] == active[tlist[:, 0]]))
    perim += np.linalg.norm(sigma_01[unique_vtx, :], axis=1).sum()
    perim += np.linalg.norm(sigma_12[unique_vtx, :], axis=1).sum()
    # 2th vertex is unique, 2!=0 and 0==1
    unique_vtx = ((active[tlist[:, 2]] != active[tlist[:, 0]])
                  & (active[tlist[:, 0]] == active[tlist[:, 1]]))
    perim += np.linalg.norm(sigma_12[unique_vtx, :], axis=1).sum()
    perim += np.linalg.norm(sigma_20[unique_vtx, :], axis=1).sum()
    perim /= 2

    # cluster stuff: in ts_vtu_to_python
    adj = sp.lil_matrix((nvtx, nvtx), dtype=bool)
    active_bonds = active[blist[:, 0]] & active[blist[:, 1]]
    adj[blist[active_bonds, 0], blist[active_bonds, 1]] = True
    _, labeled_vtx = sp.csgraph.connected_components(adj, directed=False)
    dist_size = np.bincount(np.bincount(labeled_vtx[active]))[1:]
    n_clusters = dist_size.sum()
    mean_cluster_size = (dist_size @ range(1, dist_size.size+1)
                         / n_clusters)
    std_cluster_size = (dist_size @
                        (range(1, dist_size.size+1)-mean_cluster_size)**2
                        / (n_clusters-1))
    std_cluster_size = np.sqrt(std_cluster_size)

    if w:
        v2p.write_cluster_hist(dist_size, vtu_location, v)

    if v:
        print("done with ", vtu_location)

    return (total_volume, total_area, gy_eig[0], gy_eig[1], gy_eig[2],
            nbw_nb, hmean, mean_cluster_size, std_cluster_size, perim)


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
    >$python ts_vtu_get_cluster_stat timestep_000*
    Creates pystatistics.csv:
        0, volume, area, ... linelength
        1, Volume, area, ... linelength
    >$python ts_vtu_get_cluster_stat . -o stat -w
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
                        action="store_true")
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

    stat_header = ["No", "Volume", "Area", "lamdba1", "lambda2",
                   "lambda3", "Nbw/Nb", "hbar", "mean_cluster_size",
                   "std_cluster_size", "line_length"]

    ##########################################
    # now has all vtu files. For each file:
    # Calculate and return statistics and
    # potentially write a histogram_*.csv
    # (uses the intimidately-named "multiprocessing for dummies"
    # python module instead of a for loop)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        stat_futures = executor.map(get_statistics_from_vtu,
                                    all_vtus, (v for x in all_vtus),
                                    (w for x in all_vtus))

    # debug: regular, non multiprocessing:
    # stat_futures = []
    # for vtu in all_vtus:
    #     stat_futures.append(get_statistics_from_vtu(vtu, v))

    if v:
        print("writing to main statistics file ", new_file)

    # write main statistics file
    with open(new_file, 'w', newline='') as stat_file:
        writer = csv.writer(stat_file)
        writer.writerow(stat_header)
        writer.writerows((ij[0], *ij[1]) for ij in enumerate(stat_futures))


if __name__ == "__main__":
    main()
