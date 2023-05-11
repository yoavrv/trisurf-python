#!/usr/bin/env python
# coding: utf-8
"""Wrapper for the trisurf (new cluster_trisurf version)

Has python binding to the library using CDLL and classes for the important structres (ts_vertex, ts_tape, ts_vesicle)
"""

########################
# 0: imports and paths #
########################

from ctypes import *
from pathlib import Path
import os

##############################################
# 0: Horrible homemade c parser. TODO: trash #
##############################################

def clean_text(text):
    """Clean text from /**/ and // comments, and empty lines"""
    text=text.replace('\t','    ')
    mid="/*"
    text = text
    while mid:
        pre,mid,post = text.partition('/*')
        text = pre+post.partition('*/')[-1]
    return '\n'.join(y for x in text.splitlines() if (y:=x.partition('//')[0].strip()))


def split_header_text_to_items(text):
    """Parse a cleaned header file text, return list of items ('type',['lines']) for further process.
    
    The items are of the following types:
    * compiler directive ('#',['#...'])
    * enums ('enum',['enum name {',...'last=2,}'])
    * type definitions ('typedef',['typdef void ts_void;'])
    * structure definitions ('struct',['struct ts_square {','ts_double side;','};'])
    * extern (global) ('extern',['extern ts_bool quiet;']) 
    """
    items = []
    curr = []
    open_char = ''
    for i, line in enumerate(text.splitlines()):
        # continue current item from previous open lines
        if open_char=='{':
            curr.append(line)
        # Add item: we look at #directives, enums, typedefs, structures, and externs
        if line.startswith("#"):
            curr=[line]
            items.append(('#',curr))
        if line.startswith("extern"):
            curr=[line]
            items.append(('extern',curr))
        if "enum" in line:
            curr=[line]
            items.append(('enum',curr))
        if 'typedef struct' in line:
            curr=[line]
            items.append(('typedef',curr))
            items.append(('struct',curr))
        elif 'typedef' in line:
            curr = [line]
            items.append(('typedef',curr))
        elif 'struct' in line and not open_char:
            curr=[line]
            items.append(('struct',curr))
        #
        if '{' in line:
            open_char='{'
        if '}' in  line:
            open_char=''
    return items

def text_bracket_partition(text,sep=';'):
    """Take a comment-less text of 'pre { most } post', return pre,most.split(sep),post.
    
    Useful for structs and enums, e.g.
    'enum adhesion_model { a=1,b=2,};'
    with sep=',', this will return ('enum adhesion_model',['a=1','b=2'],'')
    'typdef struct { int g; int b;} ts_coord;'
    this will return (typdef struct,['int g', 'int b'],'ts_coord')
    """
    pre, _, most = text.partition('{')
    most,_,post = most.strip().rpartition('}')
    most = most.split(sep)
    most = [y for x in most if (y := x.strip(' ;{}\n'))]
    pre = pre.strip(' \n;')
    post = post.strip(' \n;')
    return pre,most,post

def parse_line_type(line,types):
    left, _, right = line.partition(' ')
    for t in types:
        if line.startswith(t+' ') or line.startswith(t+'*'):
            _, left, right = line.partition(t)
            break
    left, right = left.strip(), right.strip()
    return left, right

def parse_line_type_inverse(line,types):
    left, _, right = line.rpartition(' ')
    for t in types:
        if t+' ' in line or t+'*' in line:
            _, left, right = line.partition(t)
            break
    return left.strip(), right.strip()

def parse_type(left, right):
    """Return """

def parse_struct(text,types):
    pre, most, post = text_bracket_partition(text)
    name = ' '.join([pre,post]).strip().split()[-1].strip(' ;')
    fields = []
    for field in most:
        if 'struct' in field:
            field = field.replace('struct','').strip()
        if not field:
            continue
        left, right = parse_line_type(field,types)
        for b in right.split(','):
            b=b.strip()
            ty=[]
            while b.startswith('*'):
                b=b[1:].strip()
                ty.append(POINTER)
            while b.endswith(']'):
                b, _, d = b[:-1].rpartition('[')
                b,d = b.strip(),d.strip()
                ty.append(lambda i,d=d: int(d)*i)
            a = types.get(left)
            for opt in ty:
                a = opt(a)
            fields.append((b,a))
    return name, fields

# mapping from c string representatin of type "int long ..." to ctype types
# it is critical that longer, more complete types come before simpler
# e.g. 'long' must be after 'long int'
base_types = {
    'signed long long int': c_longlong,    # 4 longlong
    'long long signed int': c_longlong,
    'long signed long int': c_longlong,
    'unsigned long long int': c_ulonglong, # 4 ulonglong
    'long long unsigned int': c_ulonglong,
    'long unsigned long int': c_ulonglong,
    'signed long long': c_longlong,        # 3 longlong
    'long long signed': c_longlong,
    'long signed long': c_longlong,
    'long long int' : c_longlong,  
    'unsigned long long': c_ulonglong,     # 3 ulonglong
    'long long unsigned': c_ulonglong,
    'long unsigned long': c_ulonglong,
    'signed long int': c_long,        # 3 long
    'long signed int': c_long,
    'unsigned long int': c_ulong,     # 3 ulong
    'long unsigned int': c_ulong,
    'signed short int': c_short,      # 3 short
    'short signed int': c_short,
    'unsigned short int': c_ushort,   # 3 ushort
    'short unsigned int': c_ushort,
    'signed long': c_long,      # 2 long
    'long signed': c_long,
    'long int': c_long, 
    'unsigned long': c_ulong,   # 2 ulong
    'long unsigned': c_ulong,
    'signed short': c_short,    # 2 short
    'short signed': c_short,
    'short int' : c_short,
    'unsigned short': c_ushort, # 2 ushort
    'short unsigned': c_ushort,
    'signed int': c_int,        # 2 int
    'unsigned int': c_uint,     # 2 uint       
    'signed char' : c_byte,     # 2 byte (explicit sign)
    'char signed' : c_byte,
    'unsigned char': c_ubyte,   # 2 ubyte
    'char unsigned': c_ubyte,
    'long double': c_longdouble,# 2 longdouble
    'ssize_t': c_ssize_t,
    'size_t': c_size_t,
    'float': c_float,
    'double': c_double,
    'char' : c_char,
    'int': c_int,
    'long': c_long,
    'short': c_short,
    'void' : None,
}

class TSWrapper():
    
    def __init__(self,path_to_trisurf='/opt/workspace/msc_project/cluster-trisurf'):
        self.path_to_trisurf =  Path(path_to_trisurf)
        self.path_to_trisurf_library=self.path_to_trisurf/Path('src/.libs/libtrisurf.so')
        self.path_to_trisurf_general_h =  self.path_to_trisurf/Path('src/general.h')
        if not self.path_to_trisurf_library.exists():
            raise ValueError(f"Library not found in {self.path_to_trisurf_library=}. Please update the wrapper with the right location!")
        if not self.path_to_trisurf_general_h.exists():
            raise ValueError(f"General.h not found in {self.path_to_trisurf_general_h=}. Please update the wrapper with the right location!")
        
        text = ''
        for header in self.path_to_trisurf.glob("./src/*.h"):
            with open(header,'r') as f:
                text=f'{text}\n{f.read()}'

        self.items = split_header_text_to_items(clean_text(text))
        #############################################################
        # 1: parse general.h for definitions, enums, and structures #
        #############################################################
        self.TS_SUCCESS=0
        self.TS_FAIL=1
        self.TS_ID_FILAMENT=1
        self.TS_COORD_CARTESIAN=0
        self.TS_COORD_SPHERICAL=1
        self.TS_COORD_CYLINDRICAL=2
        # defines = []
        # skip=False
        # for k,v in items:
        #     if k=='#':
                # if skip:
                #     skip=not v[0].startswith('#endif')
                #     continue
                # if v[0].startswith('#define'):
                #     defines.append((k,v[0]))
                # if v[0].startswith('#if'):
                #     skip=not v[1].endswith("_H")


        self.enums = {}
        for item, lines in self.items:
            if item=='enum':
                retext = '\n'.join(lines)
                pre,most,post=text_bracket_partition(retext,',')
                name = pre.rpartition('enum')[-1].strip()+post
                efields = {k: int(v) for k,v in map(lambda x: x.split('='),most)}
                self.enums[name]=efields
                for k,v in efields.items():
                    self.__dict__[k]=v # all enums are available in the same namespace


        self.typedefs = {}
        for item, lines in self.items:
            if item=='typedef':
                pre,_,post = text_bracket_partition('\n'.join(lines))
                defline = ' '.join((pre,post)).strip() # typedef [type...] name
                name = defline.split()[-1].strip(' ;')
                ty = ' '.join(defline.split()[1:-1]) if 'struct' not in defline else 'struct'
                self.typedefs[name]=ty

        self.ts_types={k:base_types.get(v) for k,v in self.typedefs.items()} | base_types
        for name, ty in self.ts_types.items():
            if ty is None:
                self.ts_types[name] = type(name, (Structure,),{})

        self.structs = {}
        for item in self.items:
            if item[0]=='struct':
                name, fields = parse_struct('\n'.join(item[1]),self.ts_types)
                if fields:
                    self.ts_types[name]._fields_=fields
                    self.structs[name]=fields


        for k,v in self.ts_types.items():
            self.__dict__[k]=v


        ###############################
        # 3: load the trisurf library #
        ###############################

        self.ts=CDLL(self.path_to_trisurf_library)

        # # globals
        # global_command_line_args = POINTER(ts_args).in_dll(ts,"command_line_args")
        # global_quiet = ts_bool.in_dll(ts,"quiet")
        # global_V0 = ts_double.in_dll(ts,"V0")
        # global_A0 = ts_double.in_dll(ts,"A0")
        # global_epsvol = ts_double.in_dll(ts,"epsvol")
        # global_epsarea = ts_double.in_dll(ts,"epsarea")
        self.global_vars = {}
        for item in filter(lambda x:x[0]=='extern',self.items):
            pre,most,post = text_bracket_partition("\n".join(item[1]))
            defline = ' '.join((pre,post)).strip() # typedef [type...] name
            ty = ' '.join(defline.split()[1:-1]) if 'inline' not in defline else 'inline'
            name = defline.split()[-1].strip(' ;')
            if ty!='inline' and ')' not in name:
                self.global_vars[name]=self.ts_types[ty].in_dll(self.ts,name)
                self.__dict__[name]=self.global_vars[name] # all globals are available in namespace

    ########################
    # 4: function wrappers #
    ########################

    def create_vesicle_from_tape(self,tape):
        """Using pointer for tape, it creates a vesicle, returning pointer to it."""
        self.ts.create_vesicle_from_tape.argtypes=POINTER(self.ts_tape)
        self.ts.create_vesicle_from_tape.restype=POINTER(self.ts_vesicle)
        return self.ts.create_vesicle_from_tape(tape)

    def parsetape(self,filename='tape'):
        """Loads tape with  filename (if not given it defaults to 'tape'). It returns a pointer to structure for tape"""
        self.ts.parsetape.restype=POINTER(self.ts_tape)
        self.ts.parsetape.argtypes=[c_char_p]
        return self.ts.parsetape(filename.encode('ascii'))

    def parseDump(self,filename):
        """Loads a vtu file with 'filename' and creates a vesicle returning pointer to it"""
        self.ts.parseDump.argtypes=[c_char_p]
        self.ts.parseDump.restype=POINTER(self.ts_vesicle)
        vesicle=self.ts.parseDump(filename.encode('ascii'))
        return vesicle

    def single_timestep(self,vesicle):
        """Makes a single timestep in simulations. Returns a tuple of vmsrt and bfrt (vertex move success rate and bond flip success rate)"""
        self.ts.single_timestep.argtypes=[POINTER(self.ts_vesicle),POINTER(c_double),POINTER(c_double)]
        vmsrt=c_double(0.0)
        bfsrt=c_double(0.0)
        self.ts.single_timestep(vesicle,byref(vmsrt),byref(bfsrt))
        return (vmsrt.value, bfsrt.value)

    def write_vertex_xml_file(self,vesicle,timestep_no=0):
        """Writes a vesicle into file with filename 'timestep_XXXXXX.vtu', where XXXXXX is a leading zeroed number given with timestep_no parameter (defaults to 0 if not given"""
        self.ts.write_vertex_xml_file.argtypess=[POINTER(self.ts_vesicle),self.ts_idx,POINTER(self.ts_cluster_list)]
        self.ts.write_vertex_xml_file(vesicle,self.ts_idx(timestep_no),POINTER(self.ts_cluster_list)())


    def vesicle_free(self,vesicle):
        """Free memory of the whole vesicle.

        Tape is freed seperately"""
        self.ts.vesicle_free.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.vesicle_free(vesicle)

    def vesicle_volume(self,vesicle):
        self.ts.vesicle_volume.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.vesicle_volume(vesicle)

    def vesicle_area(self,vesicle):
        ts.vesicle_area.argtypes=[POINTER(self.ts_vesicle)]
        ts.vesicle_area(vesicle)

    def gyration_eigen(self,vesicle):
        ts.gyration_eigen.argtypes=[POINTER(self.ts_vesicle), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
        l1=c_double(0.0)
        l2=c_double(0.0)
        l3=c_double(0.0)
        self.ts.gyration_eigen(vesicle , byref(l1), byref(l2), byref(l3))
        return (l1.value, l2.value, l3.value)

    def vesicle_meancurvature(self,vesicle):
        self.ts.vesicle_meancurvature.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.vesicle_meancurvature.restype=c_double
        return ts.vesicle_meancurvature(vesicle)

    def init_cluster_list(self):
        self.ts.init_cluster_list.restype=POINTER(self.ts_cluster_list)
        ret=self.ts.init_cluster_list()
        return ret

    def clusterize_vesicle(self,vesicle, cluster_list):
        self.ts.clusterize_vesicle.argtypes=[POINTER(self.ts_vesicle), POINTER(self.ts_cluster_list)]
        self.ts.clusterize_vesicle(vesicle, cluster_list)

    def cluster_list_free(self,cluster_list):
        """Free memory of cluster list"""
        self.ts.cluster_list_free.argtypes=[POINTER(self.ts_cluster_list)]
        self.ts.cluster_list_free(cluster_list)

    def stretchenergy(self,vesicle, triangle):
        self.ts.stretchenergy.argtypes=[POINTER(self.ts_vesicle), POINTER(self.ts_triangle)]
        self.ts.stretchenergy(vesicle,triangle)

    def get_absolute_ulm2(self,vesicle,l,m): # cant find this function anywhere!? maybe it's calculateKc?
        self.ts.get_absolute_ulm2.argtypes=[POINTER(self.ts_vesicle), c_double, c_double]
        self.ts.get_absolute_ulm2.restype=c_double
        ret=self.ts.get_absolute_ulm2(vesicle,l,m)
        return ret

    def getR0(self,vesicle):
        self.ts.getR0.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.getR0.restype=c_double
        r0=self.ts.getR0(vesicle)
        return r0

    def preparationSh(self,vesicle,r0):
        self.ts.preparationSh.argtypes=[POINTER(self.ts_vesicle), c_double]
        self.ts.preparationSh(vesicle,r0)

    def calculateUlmComplex(self,vesicle):
        self.ts.calculateUlmComplex.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.calculateUlmComplex(vesicle)


    def Ulm2Complex2String(self,vesicle):
        self.ts.Ulm2Complex2String.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.Ulm2Complex2String.restype=c_char_p
        string=self.ts.Ulm2Complex2String(vesicle)
        return string

    def freeUlm2String(self,string):
        self.ts.freeUlm2String.argtypes=[c_char_p]
        self.ts.freeUlm2String(string)


    #This function seems not to exist!!!
    #def solve_for_ulm2(vesicle):
    #    ts.solve_for_ulm2.argtypes=[POINTER(ts_vesicle)]
    #    ts.solve_for_ulm2(vesicle)

    def mean_curvature_and_energy(self,vesicle):
        self.ts.mean_curvature_and_energy.argtypes=[POINTER(self.ts_vesicle)]
        self.ts.mean_curvature_and_energy(vesicle)    

    def direct_force_energy(self,vesicle, vtx, vtx_old):
            """update the forces on a vertex and return the work done"""
            self.ts.direct_force_energy.argtypes=[POINTER(self.ts_vesicle),POINTER(self.ts_vertex),POINTER(self.ts_vertex)]
            self.ts.direct_force_energy.restype=c_double
            return self.ts.direct_force_energy(vesicle,vtx,vtx_old)

    def update_vertex_force(self,vesicle, vtx):
            """update the forces on a vertex using direct_force_energy"""
            self.direct_force_energy(vesicle,vtx,vtx)
            return vtx.fx, vtx.fy, vtx.fz

    def adhesion_energy_diff(self,vesicle,vtx,vtx_old):
        """Adhesion energy difference between old and new vertex"""
        self.ts.adhesion_energy_diff.argtypes=[POINTER(self.ts_vesicle),POINTER(self.ts_vertex),POINTER(self.ts_vertex)]
        self.ts.adhesion_energy_diff.restype=c_double
        return self.ts.adhesion_energy_diff(vesicle,vtx,vtx_old)

    def adhesion_geometry_distance(self,vesicle,vtx):
        """Distance between vertex and the adhesion geometry"""
        self.ts.adhesion_geometry_distance.argtypes=[POINTER(self.ts_vesicle),POINTER(self.ts_vertex)]
        self.ts.adhesion_geometry_distance.restype=c_double
        return self.ts.adhesion_geometry_distance(vesicle,vtx)

    def adhesion_geometry_side(self,vesicle,vtx):
        """is the vertex normal oriented towards the adhesion geometry"""
        self.ts.adhesion_geometry_side.argtypes=[POINTER(self.ts_vesicle),POINTER(self.ts_vertex)]
        self.ts.adhesion_geometry_side.restype=self.ts_bool
        return self.ts.adhesion_geometry_side(vesicle,vtx)

    def vertex_adhesion(self,vesicle,vtx):
        """energy of a single vertex: piece of adhesion_energy_diff"""
        delta = self.adhesion_geometry_distance(vesicle,vtx)
        is_oriented = self.adhesion_geometry_side(vesicle,vtx)
        model = vesicle.tape.adhesion_model
        dz = vesicle.tape.adhesion_cutoff
        if vtx.type&self.is_adhesive_vtx and is_oriented and delta<=dz:
            if model==self.adhesion_step_potential:
                return vtx.ad_w
            elif model==self.adhesion_parabolic_potential:
                return vtx.ad_w*(1-(delta/dz)**2)
        return 0

    def _pretty_text(self, field, thing_c):
        try:
            return (field, thing_c[:])
        except (TypeError, ValueError):
            return (field, thing_c)
 
    def pretty_text(self,thing_c):
        stuff=[]
        ret=''
        if 'contents' in dir(thing_c):
            try:
            # probably a pointer to struct
                for field, ty in thing_c.contents._fields_:
                    stuff.append(self._pretty_text(field,thing_c.contents.__getattribute__(field)))
            except ValueError:
                ret = str(thing_c)
        elif '_field_' in dir(thing_c):
            # struct
            for field, ty in thing_c._fields_:
                stuff.append(self._pretty_text(field,thing_c.__getattribute__(field)))
        else:
            ret = str(thing_c)
        if stuff:
            ret = str(thing_c) + '\n    ' +'\n    '.join(f"{x[0]} {x[1]}" for x in stuff)
        return ret

if __name__=="__main__":
    print(f"running wrapper: {ts.__name__}")