#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module for interacting with a .vtu file.

CLASSES:
    PyVtu(path): represent a trisurf .vtu file as an object 

A PyVtu object interfaces the xml as numpy array properties, with methods to
load, update, and write a new file

FUNCTIONS:
    xml_to_np(node, dtype): return node.text as numpy array
    xml_to_2_np(node, dtype): return node.text as two nx2 and mx3 arrays (for connectivity)
    np_to_xml(node, data_array): insert numpy array as node.text
    np_2_to_xml(node, blist, tlist): insert nx2 and mx3 numpy arrays as node.text (for connectivity)

more primitive functions that take .vtu as an xml.etree node and return numpy arrays
corresponding to the data in the node


Created on Thu Aug 12 14:25:18 2021

@author: yoav
"""

import re
import xml.etree.ElementTree as ET
from io import StringIO
from itertools import chain
import numpy as np


_new_debug_fields = ['mean_curvature2', 'gaussian_curvature2',
                     'new_c1', 'new_c2',
                     'eigenvalue_0', 'eigenvalue_1', 'eigenvalue_2',
                     'second_bending_modulus', 'mean_energy2',
                     'gaussian_energy2', 'normal2',
                     'eig0',  'eig1', 'eig2']
_new_interesting_fields = ['gaussian_curvature', 'gaussian_energy',
                           'bending_modulus', 'director',
                           'spontaneous_deviator', 'second_curvature'
                           'mean_curvature', 'mean_energy']
_new_vtu_fields = ['type', 'direct_force', 'adhesion_strength',
                   'force', 'normal', 'curvature']

_default_fields = ["type","c0", "force", "w", "e"] # default fields to loads (besides structural position, blist, tlist)

_known_shorthands = {"Koordinates_tock": "pos", "connectivity": ("blist","tlist"),
                     "bond_list": "blist", "triangle_list": "tlist",
                     'adhesion_strength': "ad_w", "direct_force": "f0",
                     "bonding_strength": "w", "spontaneous_curvature": "c0",
                     "curvature": "c", "bending_energy": "e",
                     "vertices_idx": "indices", "bending_modulus": "k",
                     "spontaneous_deviator": "d0",
                     "second_bending_modulus": "k2"}

# vertex types, from general.h in trisurf
vtx_type = {
            "bonding":     1, # bonding vertex, form bond with other bonding vertices
            "active":      2, #  active vertex under normally directed force
            "adhesive":    4, # adhesive vertex, subject to adhesion energy near surfaces
            "anisotropic": 8, # anisotropic vertex, requires computing full curvature characteristic
            "reserved_0": 16, # reserved type
            "vicsek":     32, # vertex under vicsek neighbor-influenced force
            "edge":       64, # edge vertex has unordered tristars
            "ghost":    -128 # ghost vertex can only be moved artificially
}

def xml_to_np(node, dtype=np.float64):
    r"""Take xml node, extract text array to numpy array of dtype.

    Convert sequence to 1d array and lines to 2d arrays, e.g
     2 3\n4 5 6\n' -> [['2','3'],['4','5','6']]
    '\n 2 3 5' -> ['2','3','5']
    """
    return np.loadtxt(StringIO(node.text), dtype=dtype)


_xml_to_2_np_pat = re.compile("\s*((?:\d+ \d+\n)*)((?:\d+ \d+ \d+\n)*)\s*")
def xml_to_2_np(node, dtype=np.int64):
    r"""Take xml node, extract text array to two numpy arrays.

    The regex pattern split the text to nx2 and mx3 arrays
    ' 1 2\n1 3\n2 3\n4 5 6\n' -> ('1 2\n1 3\n2 3\n', '4 5 6\n')
    These are then loaded using np.loadtxt to bond and triangle
    [['2','3'],['4','5','6']] -> array[[2,3]], array[[4,5,6]]
    """
    bonds, triangles = _xml_to_2_np_pat.fullmatch(node.text).groups()
    return (np.loadtxt(StringIO(bonds), dtype=dtype), np.loadtxt(StringIO(triangles), dtype=dtype))


_prefix_pat=re.compile("^(\s*)")
_postfix_pat=re.compile("(\s*)$")
def np_to_xml(node, data_array, pre=None, post=None):
    r"""Insert numpy array as text array of node.

    reverse of xml_to_np
    """
    # maintain the current format (pre- and -post spaces and tabs)
    if pre is None:
        pre = _prefix_pat.findall(node.text)[0]
    if post is None:
        post = _postfix_pat.findall(node.text)[0]

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
                     + "\n".join(" ".join(fmt(x) for x in y)
                                 for y in data_array)
                     + post)


def np_2_to_xml(node, blist, tlist):
    r"""Insert numpy array as text array of node.

    reverse of xml_to_2_np
    """
    # maintain the current format (pre- and -post spaces and tabs)
    pre = _prefix_pat.findall(node.text)[0]
    post = _postfix_pat.findall(node.text)[0]

    # choose number format based on array type
    if blist.dtype == np.float64 or tlist.dtype == np.float64:
        def fmt(x): return f"{x:.17e}"
    else:
        def fmt(x): return f"{x}"

    # insert numpy array as text: dimension matters!
    node.text = (pre
                 + "\n".join(" ".join(fmt(x) for x in y)
                             for y in chain(blist, tlist))
                 + post)


class PyVtu:
    """class representing a vtu file.

    The vtu file is parsed to an xml tree and read to numpy arrays.
    for example, access position of all vertices with 
    >>> self.pos 
    array([[ x0    , y0  , z0    ],
           [ x1    , y1  , z1    ],
           ...,
           [ x2002, y2002, zf2002]])
    access sponetaneous curvature
    >>> self.c0
    np.array([0, 0, 0, 0.5, 0, 0 ... 0])
    see .known_shorthands for list of names.

    Additional arrays can be loaded from the xml tree using .load_array 
    (and .load_data and .load_special_arrays). The.unload_array method
    removes the arrays from the object.

    The xml tree can be changed by the .update_array, update_tape, .add_array and .remove_array.
    use .write_vtu to write a new vtu file based on the new tree. 
    Only .write_vtu can change files, and will only change its own file if .write_vtu(sure=True)!
    .update_all is a convenience method to update all changes to the tree
    >>> self.w=3
    >>> # self.update_array('w') # default .write_vtu(update_all=True) does this automatically
    >>> self.write_vtu(self.path.replace('49,50'))
    *writes the next timestep_000050.vtu file modified with binding energy of 3*

    Seprate help is available on all methods.

    This is not the fastest way to manipulate vtus: to
    do things like collecting statistics from thousands of vtus,
    we should probably should use c instead
    """
    # helpful information
    known_shorthands = _known_shorthands
    default_fields = _default_fields
    vtx_type = vtx_type

    # fundemental methods
    def __init__(self, vtu_path, parse=True, load=True, load_all=True):
        """Load .vtu file

        By default, parse the xml as ElementTree tree and load all the relevant arrays
        parse=True: parse the xml file into the object
        load=True: load the geometry (position, bonds, and triangles) and default arrays
            Names are not 1-to-1 with the file! see known_shorthands for names and 
            default_fields for the fields.
        load_all=True: load all data arrays into the object
        """
        self.path = vtu_path
        self.updated = None
        self.nodes = {}
        self._arrays = {}
        if parse:
            self.tree = ET.parse(vtu_path)
            self.tape = self.tree.find("tape").text
            self.updated = False
            if load:
                self.load_special_arrays()
                self.load_data(load_all)
                self.new = self.nodes.get('type') is not None

    def load_special_arrays(self):
        """Load position, bond list, and triangle lists from the xml tree.

        Loads the special position array, which is in Points/DataArray "Koordinates tock", 
        and the bond list and triangle list, which are derived from Cell/DataArray "connectivity"
        """
        self.nodes["pos"] = self.tree.find("UnstructuredGrid/Piece/Points/DataArray")
        self.pos = xml_to_np(self.nodes["pos"],dtype=np.float64)
        self._arrays["pos"] = self.pos
        self.nodes["connectivity"] = self.tree.find("UnstructuredGrid/Piece/Cells/DataArray")
        self.blist, self.tlist = xml_to_2_np(self.nodes["connectivity"])
        self._arrays["blist"] = self.blist
        self._arrays["tlist"] = self.tlist


    def load_data(self, load_all=True, use_known_shorthand=True):
        """Load PointData and CellData from the xml tree to numpy array.

        arrays are available as self.name. Defaults to known shorthands
        _known_shorthand['curvature'] == "c0"
        otherwise uses node.attrib["Name"], replacing spaces with _
        load_all=False loads only a minimal set of arrays (listed in _default_fields)
    
        >>> self.load_all()
        >>> self.node['e'].attrib
        {'Name': 'bending energy'}
        >>> self.e
        np.array([0.986882e+1, 0.76234e+2, ...])
        """
        all_array_nodes = [*self.tree.find("UnstructuredGrid/Piece/PointData"),
                           *self.tree.find("UnstructuredGrid/Piece/CellData")]
        for node in all_array_nodes:
            if node not in self.nodes.values():
                name = node.attrib['Name'].replace(" ", "_")
                dtype=np.dtype(node.attrib['type'].lower())
                if use_known_shorthand:
                    name = self.known_shorthands.get(name, name)
                if not load_all:
                    if name not in self.default_fields:
                        continue
                self.nodes[name] = node
                self._arrays[name] = xml_to_np(self.nodes[name], dtype)
                self.__dict__[name] = self._arrays[name]

    def _find_node(self, node_name) -> ET.Element:
        """Find and return node."""
        return self.tree.find(f'.//*[@Name=\"{node_name}\"]')

    def load_array(self, name: str, vtu_array_name: str = None, raise_error=False)->np.array:
        """Load (or reload) an array from the xml tree into the PyVtu object.

        Array is available as self.name. 
        Optionally load vtu_array_name but make it available as self.name.
        undo with .unload_array(name)

        >>> self.load_array("w","bonding_strength")
        >>> self.w
        array([2., 2., 2., ..., 2., 2., 2.])
        """
        if vtu_array_name is None:
            vtu_array_name = name
        if name not in self.__dict__: # node is unknown
            node = self._find_node(vtu_array_name)
            if node is None:
                if raise_error:
                    raise KeyError(f"{vtu_array_name} could not be found in xml tree")
                return None
            else:
                if node in self.nodes.values():
                    if raise_error:
                        name_node = next(k for k, v in self.nodes.items() if v==node)
                        raise AttributeError(f"{name} already exists as {name_node}")
                self.nodes[name] = node
                self._arrays[name] = xml_to_np(node)
                self.__dict__[name] = self._arrays[name]
                return self.__dict__[name]
        else:
            if name in self.nodes:
                self._arrays[name] = xml_to_np(self.nodes[name], self._arrays[name])
                self.__dict__[name] = self._arrays[name]
            elif name in {"blist", "tlist"}:
                self.blist, self.tlist = np_2_to_xml(self.nodes["connectivity"])
                self._arrays['blist'] = self.blist
                self._arrays['tlist'] = self.tlist
            return self.__dict__[name]


    def unload_array(self, name)->ET.Element:
        """Unload (drop) array from object.

        Opposite of load_array. returns the XML element (not removed from tree)
        raise error if there is no such array.

        >>> self.unload_array('c0')
        >>> self.c0
        AttributeError: 'PyVtu' object has no attribute 'c0'
        """
        if name in self.__dict__:
            self.__dict__.pop(name)
            self._arrays.pop(name)
            return self.nodes.pop(name)
        else:
            raise AttributeError(f"{name} does not exist or not loaded!")


    def update_array(self, name)->ET.Element:
        """Update a loaded array into the xml tree.

        name must be a known (a key in self.nodes or 'tlist'/'blist'), and was updated in
        self.__dict__ (by self.name = new_array)
        """
        if name in self.nodes:
            self._arrays[name] = self.__dict__[name]
            np_to_xml(self.nodes[name], self._arrays[name])
            self.updated = True
            return self.nodes[name]
        elif name in {"blist", "tlist"}:
            self.blist = self.__dict__['blist']
            self.tlist = self.__dict__['tlist']
            self._arrays['blist'] = self.__dict__['blist']
            self._arrays['tlist'] = self.__dict__['tlist']
            np_2_to_xml(self.nodes["connectivity"], self.blist, self.tlist)
            self.updated = True
            return self.nodes['connectivity']
        else:
            raise AttributeError(f'{name} is not loaded (no associated node)')

    def add_array(self, name, array=None,
                     vtu_array_name=None, parent='PointData')->ET.Element:
        """Add an array into the xml tree.

        Update existing array if name already exist in object.
        Find or create new element in tree if it does not.

        >>> self.add_array('too_high',self.pos[:,2]>0,'Too High')
        >>> self.too_high
        [True, False, False, True...]
        >>> ET.tostring(self.nodes['too_high'])
        <DataArray, type="Int64" Name="Too High" format="ascii">1 0 0 1 ... </DataArray>

        Does not write into the file! use write_vtu to write the tree
        into a real .vtu file.
        """
        if name in self._arrays:
            if array is not None:
                if np.shape(array)==np.shape(self.__dict__[name]):
                    self._arrays[name] = array
                    self.__dict__[name] = array
                else:
                    # wrong shape, e.g. self.w=2 needs to update the array to [2,2,2,2...]
                    self._arrays[name][:] = array
                    self.__dict__[name] = self._arrays[name]
            return self.update_array(name)
        if array is None:
            raise ValueError(f'Added None instead of array to {name}')

        if type(parent) is str:
            parent = self.tree.find(f".//*{parent}")
        if vtu_array_name is None:
            vtu_array_name = name
        if array.dtype == np.dtype(float):
            _type = 'Float64'
        else:
            _type = 'Int64'
        attrib = {'type': _type, 'Name': vtu_array_name}
        if len(array.shape) == 2:
            attrib['NumberOfComponents'] = str(int(array.shape[1]))
            post = '\n'
        else:
            post = ' '
        attrib['format'] = 'ascii'
        # find if element exists, else make new one
        for sub_element in parent:
            if sub_element.attrib == attrib:
                node = sub_element
                break
        else:  # couldn't find node: create one
            node = ET.SubElement(parent, 'DataArray', attrib)
        node.tail = parent.tail  # '\n'
        np_to_xml(node, array, pre='', post=post)
        self.nodes[name] = node
        self._arrays[name] = array
        self.__dict__[name] = self._arrays[name]
        self.updated = True
        return node

    def update_tape(self, text=None, replace=True):
        """Update tape in tree. if text is not provided, self.tape is used."""
        if text is None:
            self.tree.find("tape").text = self.tape
        else:
            self.tree.find("tape").text = text
            if replace:
                self.tape = text
        self.updated=True

    def update_all(self):
        """Update all arrays and tape into the tree.
        
        Usefull for changes such as self.w=2
        """
        self.update_tape()
        for array_name in self._arrays:
            if array_name != "tlist":  # updated along with blist
                self.update_array(array_name)

    def _remove_from_tree(self, node):
        """Remove node from xml tree.

        Return element on success, raise error if can't find node.
        """
        if node is None:
            return
        try:
            pd = self.tree.find(".//*PointData")
            return pd.remove(node)
        except ValueError:
            try:
                cd = self.tree.find(".//*CellData")
                return cd.remove(node)
            except ValueError:
                raise AttributeError(
                    f'{node} {node.attrib["Name"]} is'
                    ' not a leaf of PointData or CellData')

    def remove_node(self, name, vtu_array_name=None):
        """Remove node from tree.

        Takes name (such as 'w') and remove from the vtu
        this removes it from the tree when vtu is written
        use vtu_array_name to directly remove a node if it does not have
        an associated array and name ('bonding_strength')
        """
        if vtu_array_name is None:
            vtu_array_name = name
        if name in self.__dict__:
            node = self.nodes.pop(name)
            self.__dict__.pop(name)
            self._arrays.pop(name)
        else:
            node = self._find_node(vtu_array_name)
        self._remove_from_tree(node)
        self.updated = True

    def remove_array(self, name):
        """Remove array from node and node from xml tree.
        
        Opposite of add_array"""
        node = self.unload_array(name)
        self._remove_from_tree(node)

    def write_vtu(self, file_path, sure=False, update_all=True):
        """Write a new vtu with updated arrays."""
        if not self.updated:
            raise RuntimeError("nothing new to write!")
        if file_path == self.path and not sure:
            raise RuntimeError("Attempted to override vtu when"
                               " sure=False (not sure)")
        if update_all:
            self.update_all()
        self.tree.write(file_path)

    # utility methods

    def remove_new_fields(self, strict=False, superstrict=False):
        """Remove boring, useless fields added by new versions.

        strict: remove boring modeled_trisurf addition and other debug ones
        superstrict: remove all newer addition to trisurf
        """
        names = _new_debug_fields
        if strict:
            names += _new_interesting_fields
            if superstrict:
                names += _new_vtu_fields
        for name in names:
            try:
                self.remove_node(name)
            except AttributeError:
                print(f"couldn't find {name}")

    def get_neighbors(self, vertex: int) -> np.array:
        """Array of the indices of the vertex neighbors."""
        return self.blist[(self.blist == vertex)[:, ::-1]]

    def get_ordered_neighbors(self, vertex: int) -> np.array:
        """Array of the indices of the vertex neighbors, ordered by triangles."""
        v = self.tlist == vertex  # triangles containing vertex
        tria = self.tlist[v.any(axis=1), :]  # reduce to the relevent triangles
        v = v[v.any(axis=1), :]
        # triangle = {this, left, right} + permutations
        firsts = np.zeros_like(v)  # get left
        firsts[v[:, 0], 1] = True
        firsts[v[:, 1], 2] = True
        firsts[v[:, 2], 0] = True
        left = tria[firsts]
        seconds = np.zeros_like(v)  # get right
        seconds[v[:, 0], 2] = True
        seconds[v[:, 1], 0] = True
        seconds[v[:, 2], 1] = True
        right = tria[seconds]
        nei = np.zeros_like(right)  # neighbor output
        # build neighbors left to right, "bouncing" between the arrays
        # left[next_idx++] == right[next_idx]
        # left=[a,d,b,c], right=[b,a,c,d] => next_idx=0->2->3->1
        next_idx = 0
        for i, _ in enumerate(left):
            nei[i] = left[next_idx]
            next_idx = np.where(left == right[next_idx])[0][0]
        return nei

    def str_pos(self, vertex: int):
        """Return string representation x,y,z of vertex position."""
        return (f"{self.pos[vertex,0]},"
                f"{self.pos[vertex,1]},"
                f"{self.pos[vertex,2]}")

    def v_to_mat(self, vertex: int) -> str:
        """Print copy-pastable vertex to mathematica list."""
        return f"{{ {self.str_pos(vertex)} }}"

    def v_nei_to_mat(self, vertex: int) -> str:
        """Print copy-pastable vertex neighbors to mathematica list."""
        return ("{{"
                + "},{".join(self.str_pos(x)
                             for x in self.get_ordered_neighbors(vertex))
                + "}}")
    


def vertex_to_mathematica(py_vtu, vertex: int):
    """Print copy-pastable vertex and neighbors to mathematica."""
    return (f"{{ {py_vtu.str_pos(vertex)} }}",
            "{{"
            + "},{".join(py_vtu.str_pos(x)
                         for x in py_vtu.get_ordered_neighbors(vertex))
            + "}}")


def print_many(my_vtu: PyVtu, vertices: list, ext_label: str = ""):
    """Print points and ordered neighbors for a mathematica scripts."""
    points = [my_vtu.v_to_mat(x) for x in vertices]
    neighborhoods = [my_vtu.v_nei_to_mat(x) for x in vertices]
    print(f"points{ext_label}={{", ",".join(points), "}")
    print(f"neis{ext_label}={{", ",".join(neighborhoods), "}")


def reconstruct_shape_operator(v: PyVtu) -> np.array:
        """Reconstruct the shape tensor from eigenvectors and eigenvalues"""
        g=np.stack((v.eig0,v.eig1,v.eig2),axis=1)
        s=np.zeros(g.shape)
        s[:,0,0]=v.eigenvalue_0
        s[:,1,1]=v.eigenvalue_1
        s[:,2,2]=v.eigenvalue_2
        return np.einsum("nij,njk,nlk->nil",g,s,g)


def get_nematic_order_parameter(v:PyVtu, update_vtu=False):
    """Get the nematic order for each vertex in v.
    
    For an anisotropic vertex i (v.type[i]&8!=0), the nematic order parameter is
    S = 0.5*(3 director_i@director_j-1) averaged over all anisotropic neighbors j.
    For non anisotropic vertices, S=0
    """
    t_ani = vtx_type['anisotropic']
    s = 0.0*v.type
    num_nei = 0*v.type
    blist = v.blist[(v.type[v.blist]&t_ani!=0).all(axis=1)]
    ninj = (v.director[blist[:,0]]*v.director[blist[:,1]]).sum(axis=1)
    for nn, bond in zip(ninj, blist):
        s[bond[0]]+=0.5*(3*nn-1)
        s[bond[1]]+=0.5*(3*nn-1)
        num_nei[bond[0]]+=1
        num_nei[bond[1]]+=1
    s/=(num_nei+num_nei==0) # average neighbors
    if update_vtu:
        v.add_array('s',s,'Nematic_Order')
    return s


_rmfields = {'debug':_new_debug_fields,
             'reduce_new': _new_debug_fields + _new_interesting_fields,
             'all': _new_debug_fields + _new_interesting_fields + _new_vtu_fields}
def _xmloctomy(vtu_path, streamline=False, rmfields=None):
    """Remove the <trisurf> tag of a .vtu file (dangerous!), remove tape info, and some of the fields.

    This means the file cannot be used to recreate the simulation!

    rmfields='debug' will remove various debug fields (like eig0, eig1)
    rmfields='reduce_new' will remove more fields (like gaussian curvature)
    rmfields='all' will remove almost all new fields (like type)
    rmfields=['field1',...] will remove the specified fields (by element name!)
    """
    tree = ET.parse(vtu_path)
    root = tree.getroot()
    did_something = False
    tri_tag = root.find("trisurf")
    if tri_tag is not None:
        root.remove(tri_tag)
        did_something = True
    if streamline:
        tape = root.find("tape")
        tape.text = _streamline_tape(tape.text)
        did_something = True
    if rmfields is not None:
        try:
            rmfields = _rmfields[rmfields]
        except (TypeError,KeyError):
            pass
        point_array_node = tree.find("UnstructuredGrid/Piece/PointData")
        cell_array_node = tree.find("UnstructuredGrid/Piece/CellData")
        for node in list(point_array_node):
            if node.attrib['Name'] in rmfields:
                did_something = True
                point_array_node.remove(node)
        for node in list(cell_array_node):
            if node.attrib['Name'] in rmfields:
                did_something = True
                cell_array_node.remove(node)
    if did_something:
        tree.write(vtu_path)


def _streamline_tape(text):
    """Chop and throw out any irrelevent part of the tape."""
    no_care = {'dmax', 'dmin_interspecies', 'poly', 'nmono', 'k_spring',
               'internal_poly', 'nfil', 'nfono', 'xi',
               'spherical_harmonics_coefficients', 'quiet', 'multiprocessing',
               'smp_cores', 'cluster_nodes', 'distributed_processes'}
    return "\n".join(x for x in text.splitlines()
                     if len(x) > 0 and x[0] != '#' and x[0] != ' '
                     and x.split('=')[0] not in no_care)


_arrays_to_change_type = {'c0','d0','f0','w', 'ad_w','type', 'k', 'k2'}
def change_vertices(v: PyVtu, indices, type_, values=None):
    """Change of vertices 'indices' to the new 'type_', by values dictionary/first of the same type.
    
    values should be a dictionary {'c0': ,'d0': ,'f0': ,'w': , 'ad_w': ,'type': , 'k': , 'k2': } 
    updating each value, or None, in which case the first vertex in 'v' which is of the new type
    is assumed to be the prototype for the rest.
    """
    if not values:
        pro = v.indices[(v.type==type_)][0]
        values = {key: v._arrays[key][pro] for key in _arrays_to_change_type if key in v._arrays}
    for array, value in values.items():
        if array in v._arrays:
            v._arrays[array][indices]=value
        else:
            raise KeyError(f"{array} is not found in v and can't be updated!")

_adhesion_z= re.compile('^(z_adhesion|adhesion_z|adhesion_cuttoff|adhesion_cutoff)=(.*)',re.MULTILINE)
def saddle_vtu_indices(v: PyVtu, num=50, direction: tuple=(1,0,0)):
    """Return the indices of vertices that would "saddle" the vtu (leading edge)."""
    z_ad, cut = _adhesion_z.findall(v.tape)
    if 'z' not in z_ad[0]:
        z, dz = float(cut[1]), float(z_ad[1])
    else:
        z, dz = float(z_ad[1]), float(cut[1])
    bottom = v.pos[:,2]<(z+dz)
    left, right = v.blist[:,0], v.blist[:,1]
    edge_bond = bottom[left]^bottom[right]
    edge = np.isin(v.indices,np.unique(edge_bond))
    return v.indices[edge][(v.pos[edge]@direction).argsort()[-num:]]

