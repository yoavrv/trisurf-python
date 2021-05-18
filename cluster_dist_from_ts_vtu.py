#!/usr/bin/env python
# coding: utf-8
"""Obtain statistics of cluster size on .vtu files produced by trisurf.

Takes directories and list of .vtu files, and create histograms of the
cluster size distribution for all .vtu files in them. The histograms
are named the same affixed with "histogram_", unless they contain "timestep",
In which case the "timestep" is replaced with "histogram"

Example: suppose /mnt/c/paraview_storage contained "Nth_sim.vtu" with N=1...10
and the current directory contains "timestep_0123456.vtu",
Running in the command line
   $python ts_vtu_cluster_stat.py /mnt/c/paraview_storage *.vtu
would create 10 histogram files in /mnt/c/paraview_storage
named "histogram_Nth_sim.csv"
and 1 histogram file in the current directory
named "histogram_0123456.vtu"

"""

# imports
import ts_vtu_to_python as v2p
import os
import concurrent.futures
import argparse
import csv


def write_cluster_hist_from_vtu(vtu_location, v):
    """Subfunction to main, write the histogram file for each vtu file.

    Takes vtu_location, the path of the vtu file, and v, the verbosity flag
    calculates and writes an appropriate histogram file
    If vtu_location is "/over/here/blah.vtu"
    creates "/over/here/histogram_blah.csv"
    Special case: of name contains "timestep". replace
    i.e. if vtu_location is "/over/here/timestep_01.vtu"
    creates "/over/here/histogram_01.csv"
    """
    # get cluster distribution:
    clusters = v2p.cluster_dist_from_vtu(vtu_location)
    if v:
        print("extracted ", clusters[:5])

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
        print('writing ', clusters[:5], '... to ', hist)

    # write .csv file
    with open(hist, 'w', newline='') as hist_file:
        writer = csv.writer(hist_file)
        writer.writerow(['cluster_size', 'number_of_clusters'])
        # write only rows with nonzero clusters
        relevant_rows = (x for x in zip(range(1, len(clusters)+1), clusters)
                         if x[1] != 0)
        writer.writerows(relevant_rows)


def main():
    """Parse command line args to find .vtu files, create cluster histogram.

    See argparse description for how to use.
    Example use:
    >$python ts_vtu_get_cluster_stat .
    All .vtu in current directory get a histogram file
    >$python ts_vtu_get_cluster_stat /vtu_cache
    All .vtu files in /vtu_cache directory get a histogram file
    >$python ts_vtu_get_cluster_stat ./my_favorite_vtu.vtu
    Creates ./histogram_my_favorite_vtu.csv file
    cluster_size, number_of_clusters
    1, 2013
    2,56
    ...
    etc
    >$python ts_vtu_get_cluster_stat timestep_000*
    Creates historgam_000*.vtu files
    """
    # parse the arguments:
    parser = argparse.ArgumentParser(
        description='''Get cluster size statistics for all .vtu files, listed
        or from directories, and write them to histogram_[vtu_name]_.csv files.
        If a name contains "timestep", it is replaced with "histogram" instead.
        ''')
    parser.add_argument("vtu_in", nargs='+',
                        help='.vtu files or directory with .vtu files')

    parser.add_argument("-v", "--verbose", help='increase output verbosity',
                        action="store_true")
    args = parser.parse_args()

    v = args.verbose  # ease verbosity checks

    # make the files input iterable, even if it's one
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
            print('got ', vtus)

        all_vtus.extend(vtus)

    # now has all vtu files, now for each file:
    # Calculate cluster size distribution and
    # write a histogram_*.csv
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(write_cluster_hist_from_vtu,
                     all_vtus, (v for x in all_vtus))


if __name__ == "__main__":
    main()
