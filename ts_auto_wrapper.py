#!/usr/bin/env python
# coding: utf-8
"""Automatic Wrapper for the trisurf 

use
>>> from ts_auto_wrapper import TSWrapper
>>> ts = TSWrapper('/path/to/trisurf/project/trisurf_ng')
(see documentation on TSWrapper)

The wrapper is rudimentary, and cannot parse complex, bracketed, or recurvsive definitions,
especially function pointers.
The wrapper also requires foreknowledge on the types used by the libraries, such as
'gsl_complex' and 'xmlDocPtr'. in order to parse the headers. 
This is handled by the dictionaries 'base_types' and 'added_types', map the names to ctypes types.
TSWrapper can also take a dictionary for even more types.
"""

from ctypes import *
from pathlib import Path
import sys
import warnings

##############################################
# 0: Horrible homemade c parser. TODO: trash #
##############################################

# mapping from c string representation of type "int long ..." to ctype types
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

added_types = {
    'gsl_complex' : type('gsl_complex', (Structure,),{'_fields_':[('dat',2*c_double)]}),
    'xmlDocPtr' : c_void_p,
    'xmlNodePtr' : c_void_p,
}

def clean_text(text):
    """Clean text from /**/ and // comments, and empty lines."""
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
    * extern (globals) ('extern',['extern ts_bool quiet;']) 
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
    >>> text_bracket_partition('enum adhesion_model { a=1,b=2,};', sep=',')
    ('enum adhesion_model', ['a=1','b=2'], '')
    >>> text_bracket_partition('typdef struct { int g; int b;} ts_coord;')
    ('typdef struct', ['int g', 'int b'], 'ts_coord')
    """
    pre, _, most = text.partition('{')
    most,_,post = most.strip().rpartition('}')
    most = most.split(sep)
    most = [y for x in most if (y := x.strip(' ;{}\n'))]
    pre = pre.strip(' \n;')
    post = post.strip(' \n;')
    return pre,most,post


def parse_type(line,types):
    """Parse a general type declaration 'type0 type1 *a,b,c[];' to base_type and name-type doctionray.
    
    Rudimentary: this is not capable of parsing parenthesis (especially function pointer) 
    or multi-level declarations.
    >>> parse_type('long int*unsigned, b[3], c', base_types)
    (long int, {'unsigned': ctypes.LP_c_long, 'b': ctypes.c_long_Array_3, 'c': ctypes.c_long})
    """
    first, *rest = line.replace('*',' * ').strip(' ;').split(',')
    base_type = first.replace('*','').replace('[','').replace(']','').replace(';','').rpartition(' ')[0].strip()
    left = first.replace(base_type,'').strip()
    res = [left, *rest]
    out = {}
    for item in res:
        if not item.strip():
            continue # skip excess non-fields 'int a,'->'a',''
        ty = types.get(base_type)
        num_star=item.count('*')
        item=item.replace('*','').strip()
        for i in range(num_star):
            ty = POINTER(ty)
        while ']' in item:
            item, _, _ = item.rpartition(']')
            item, _, num = item.rpartition('[')
            ty = int(num)*ty if num else POINTER(ty)
        out[item.strip()]=ty
    return base_type, out

def parse_struct(text,types):
    """Parse a struct "... s_name {ty0 field0; ty1 field1;...}" to name and fields [(name, type)].
    
    This is used for constructing ctypes Structures._fields_
    e.g.
    >>> parse_struct('typdef struct { int g; int b;} ts_coord;',base_types)
    ('ts_coord', [('g', ctypes.c_int),('b', ctypes.c_int)])
    """
    pre, most, post = text_bracket_partition(text)
    name = ' '.join([pre,post]).strip().split()[-1].strip(' ;')
    fields = []
    for field in most:
        if 'struct' in field:
            field = field.replace('struct','').strip()
        if not field:
            continue
        _, out = parse_type(field,types)
        for item_name, item_type in out.items():
            fields.append((item_name, item_type))
    return name, fields


class TSWrapper():
    """Class that instantiate a trisurf wrapper from a path.
    
    Design to act like a module i.e.
    >>> from ts_auto_wrapper import TSWrapper
    >>> ts = TSWrapper('/path/to/trisurf/project/trisurf_ng')
    Exposes python binding to the library using CDLL.
    Everything is available in ts.X, but they are also organized by types with faster autocompletes:
        ts.ts_types: classes for types (ts_vertex, ts_tape, ts_vesicle, ...)
        ts.functions: functions (init_vertex, parseDump, ...)
        ts.globals: global variables
        ts.enums: enum definitions
        ts.TS_...: several #defines are hardcoded (e.g. TS_SUCCESS)
    and misc. things: ctype function POINTER, pointer; a pretty_print, and a byte_to_int function

    The raw functions with 'restype' and 'argtype' are available as in ts.functions._c_functionname
    The CDLL itself is available in ts.cdll

    The wrapper also requires knowledge on the names of types used by the libraries, such as
    'gsl_complex' and 'xmlDocPtr'. TSWrapper can accept a dictionary for more types {'name': ctypes.structure...}.
    """
    
    def __init__(self,path_to_trisurf='/opt/workspace/msc_project/cluster-trisurf', more_types=None):
        """Initializing the wrapper: Roughly the equivalent of a wrapper module code, indented twice."""
        
        # add module-like functionality:
        self.clean_text = clean_text
        self.base_types = base_types
        self.added_types = added_types
        if more_types:
            self.added_types = added_types.update(more_types)
        self.split_header_text_to_items = split_header_text_to_items
        self.text_bracket_partition = text_bracket_partition
        self.parse_struct = parse_struct
        self.POINTER = POINTER
        self.pointer = pointer
        
        # trace path to trisurf and attempt to load it
        self.path_to_trisurf = Path(path_to_trisurf)
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
        self.text=clean_text(text)
        self.items = split_header_text_to_items(self.text)
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
        
        # use typedef to get types

        self.typedefs = {}
        for item, lines in self.items:
            if item=='typedef':
                pre,_,post = text_bracket_partition('\n'.join(lines))
                defline = ' '.join((pre,post)).strip() # typedef [type...] name
                name = defline.split()[-1].strip(' ;')
                ty = ' '.join(defline.split()[1:-1]) if 'struct' not in defline else 'struct'
                self.typedefs[name]=ty

        self.ts_types= ({k: self.base_types.get(v) for k,v in self.typedefs.items()} 
                        | base_types
                        | added_types
                       )
        
        # structures: initialize per type
 
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

        self.cdll=CDLL(self.path_to_trisurf_library)

        # # globals
        self.globals = {}
        for item in filter(lambda x:x[0]=='extern',self.items):
            pre,most,post = text_bracket_partition("\n".join(item[1]))
            defline = ' '.join((pre,post)).strip() # typedef [type...] name
            ty = ' '.join(defline.split()[1:-1]) if 'inline' not in defline else 'inline'
            name = defline.split()[-1].strip(' ;')
            if ty!='inline' and ')' not in name:
                self.globals[name]=self.ts_types[ty].in_dll(self.cdll,name)
                self.__dict__[name]=self.globals[name] # all globals are available in namespace

        self.dec_funcs = {}
        text=(self.text.replace(' extern ',' ').replace(' inline ',' ').replace(' const ',' ')
                       .replace('\nextern ','\n').replace('\ninline ','\n').replace('\nconst ','\n')
                       .replace(';extern ',';').replace(';inline ',';').replace(';const ',';')
                       .replace('(extern ','(').replace('(inline ','(').replace('(const ','('))
        if '(*' in text:
            warnings.warn('detected possible strange pointer (*.... Yoav isn\'t smart enough to parse that!')
        else:
            while '(' in text:
                pre,_,post = text.partition('(')
                args, _, post = post.partition(')')
                text = post.strip()
                if '(' in args:
                    errstr = f'{line_start}({args}){text.partition(" ")[0]}'
                    raise ValueError(f'{errstr} detected "(" in candidate argument list {args}. Yoav isn\'t smart enough to parse this')
                line_start = pre.splitlines()[-1]
                if len(line_start)>2 and '#' not in line_start[0]:
                    tyname, line = parse_type(line_start,self.ts_types)
                    name, ty = line.popitem()
                    if text.startswith('['):
                        errstr = f'{line_start}({args}){text.partition(" ")[0]}'
                        raise ValueError(f'{errstr} found "type stuff(...)[...". Yoav doesn\'t know how to parse that')
                    if line:
                        errstr = f'{line_start}({args}){text.partition(" ")[0]}'
                        raise ValueError(f'{errstr} parsed an extra {line} item on top of {name}:{ty}')
                    self.dec_funcs[name]=(tyname,ty,args)
        
        class ts_functions: 
            def __getitem__(self,s):
                return self.__getattribute__(s)
        self.functions=ts_functions()
        ts_success = self.ts_types['ts_bool'](self.TS_SUCCESS).value
        ts_fail = self.ts_types['ts_bool'](self.TS_FAIL).value
        def process_str_to_char_p(arg): return str(arg).encode('ascii');
        def process_true_false_to_ts_bool(arg): return ts_success if arg else ts_fail;
        def process_create_double_ref(arg): 
            return pointer(c_double(arg)) if type(arg) in {int, float} else arg;
        def process_neutral(arg): return arg;
        def process_ts_bool_to_true_false(arg): return arg==ts_success;
        def process_double_ref_to_doube(arg): return arg.contents.value;

        try:
            for func, (tyname, ty, args) in self.dec_funcs.items():
                signature=[parse_type(y,self.ts_types) for x in args.split(',') if (y:=x.strip())]
                signature_name_type = [list(x[1].items())[0] for x in signature]
                signature_type = [x[1] for x in signature_name_type]
                self.functions.__dict__["_c_"+func] = _c_f = self.cdll[func]
                self.functions.__dict__["_c_"+func].argtype=signature_type
                self.functions.__dict__["_c_"+func].restype=ty
                signature_str= ""
                process_args=[]
                process_out=[]
                arg_types = []
                if tyname=='ts_bool':
                    process_out.append(process_ts_bool_to_true_false)
                    ret_type_str = f"Return {tyname} as True/False"
                else:
                     process_out.append(process_neutral)
                     ret_type_str = f"Return {ty}"
                for (arg_type_name, _), (arg_name,arg_type) in zip(signature,signature_name_type):
                    if arg_type_name=='ts_bool':
                        process_args.append(process_true_false_to_ts_bool)
                        process_out.append(None)
                        arg_types.append(arg_type)
                        signature_str = f'{signature_str}, {arg_name}: bool'
                    elif arg_type==c_char_p or arg_type==POINTER(c_char):
                        process_args.append(process_str_to_char_p)
                        process_out.append(None)
                        arg_types.append(arg_type)
                        signature_str = f'{signature_str}, {arg_name}: str or __str__'
                    elif arg_type==POINTER(c_double):
                        process_args.append(process_create_double_ref)
                        process_out.append(process_double_ref_to_doube)
                        arg_types.append(arg_type)
                        signature_str = f'{signature_str}, {arg_name}: {arg_type} or double'
                        ret_type_str = f"{ret_type_str}, {arg_name}: double"
                    else:
                        process_args.append(process_neutral)
                        process_out.append(None)
                        arg_types.append(arg_type)
                        signature_str = f'{signature_str}, {arg_name}: {arg_type}'

                doc = f"Wrapper for {func} with arguments {signature_str}. {ret_type_str}"
                def f(*args, _base_c_function=_c_f, arg_types=arg_types,
                       process_args=process_args, process_out=process_out):
                    nargs = [f(a) for f,a in zip(process_args,args)]
                    l=len(nargs)
                    nargs.extend([f(0) for f,t in zip(process_args[l:],arg_types[l:])
                                  if t==POINTER(c_double)])
                    if len(nargs)<len(arg_types):
                        raise ValueError(f"Not enough arguments: given {args}, proccessed to {nargs}, requires {arg_types}")
                    ret = _base_c_function(*nargs)
                    all_ret = (*[f(x) for f,x in zip(process_out,(ret,*nargs)) if f],)
                    if len(all_ret)==1:
                        return all_ret[0]
                    return all_ret
                f.__doc__=doc
                self.functions.__dict__[func]=f
                self.__dict__[func]=f
        except (RuntimeError,IndexError,ValueError,KeyError) as e:
            warnings.warn("Error while trying to parse functions due to the following error:")
            print(e)
    

    ################################
    # 4: special function wrappers #
    ################################


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
        """Nicer string representation of a ctypes struct, like LP_ts_tape"""
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
    
    def byte_to_int(self,byte):
        return int.from_bytes(byte,sys.byteorder)


if __name__=="__main__":
    print(f"running wrapper: {__name__}")
    ts = TSWrapper()
