from ctypes import *

TS_SUCCESS=0
TS_FAIL=1

TS_ID_FILAMENT=1

TS_COORD_CARTESIAN=0
TS_COORD_SPHERICAL=1
TS_COORD_CYLINDRICAL=2


class ts_coord(Structure):
	_fields_=[
		("e1", c_double),
		("e2", c_double),
		("e3", c_double),
		("coord_type", c_uint)
		]
class ts_vertex(Structure):
	pass
class ts_bond(Structure):
	pass
class ts_triangle(Structure):
	pass
class ts_cell(Structure):
	pass
class ts_poly(Structure):
	pass
class ts_cluster(Structure):
	pass
ts_vertex._fields_=[
		('idx',c_uint),
		('x',c_double),
		('y',c_double),
		('z',c_double),
		('neigh_no',c_uint),
		('neigh', POINTER(POINTER(ts_vertex))),
		('bond_length', POINTER(c_double)),
		('bond_length_dual',POINTER(c_double)),
		('curvature', c_double),
		('energy', c_double),
		('energy_h',c_double),
		('tristar_no', c_uint),
		('tristar', POINTER(POINTER(ts_triangle))),
		('bond_no',c_uint),
		('bond',POINTER(POINTER(ts_bond))),
		('cell',POINTER(ts_cell)),
		('xk',c_double),
		('c',c_double),
		('id', c_uint),
		('projArea',c_double),
		('relR', c_double),
		('solAngle', c_double),
		('grafted_poly', POINTER(ts_poly)),
		('cluster',POINTER(ts_cluster)),
		]
class ts_vertex_list(Structure):
	_fields_=[('n',c_uint), ('vtx',POINTER(POINTER(ts_vertex)))]

ts_bond._fields_=[('idx',c_uint),
		('vtx1', POINTER(ts_vertex)),
		('vtx2', POINTER(ts_vertex)),
		('bond_length',c_double),
		('bond_length_dual',c_double),
		('tainted', c_char),
		('energy',c_double),
		('x',c_double),
		('y',c_double),
		('z',c_double),
	]
class ts_bond_list(Structure):
	_fields_=[('n', c_uint),('bond',POINTER(POINTER(ts_bond)))]

ts_triangle._fields_=[
		('idx',c_uint),
		('vertex', POINTER(ts_vertex)*3),
		('neigh_no',c_uint),
		('neigh', POINTER(POINTER(ts_triangle))),
		('xnorm', c_double),
		('ynorm', c_double),
		('znorm', c_double),
		('area', c_double),
		('volume', c_double),
		('energy', c_double),
	]

class ts_triangle_list(Structure):
	_fields_=[('n',c_uint),('a0',c_double),('tria', POINTER(POINTER(ts_triangle))),]


ts_cell._fields_=[
	('idx', c_uint),
	('vertex', POINTER(POINTER(ts_vertex))),
	('nvertex', c_uint),
	]		

class ts_cell_list(Structure):
	_fields_=[
		('ncmax', c_uint*3),
		('cellno', c_uint),
		('cell',POINTER(POINTER(ts_cell))),
		('dcell', c_double),
		('shift', c_double),
		('max_occupancy', c_double),
		('dmin_interspecies', c_double)
	]

class ts_spharm(Structure):
	_fields_=[
		('l',c_uint),
		('ulm', POINTER(POINTER(c_double))),
	#	('ulmComplex', POINTER(POINTER(gsl_complex))), #poisci!!!!
		('ulmComplex', POINTER(POINTER(c_double))), #temporary solution
		('sumUlm2', POINTER(POINTER(c_double))),
		('N', c_uint),
		('co',POINTER(POINTER(c_double))),
		('Ylmi', POINTER(POINTER(c_double))),
		]

ts_poly._fields_=[
		('vlist', POINTER(ts_vertex_list)),
		('blist', POINTER(ts_bond_list)),
		('grafted_vtx',POINTER(ts_vertex)),
		('k', c_double),
	]

class ts_poly_list(Structure):
	_fields_=[('n',c_uint),('poly',POINTER(POINTER(ts_poly)))]

class ts_tape(Structure):
	_fields_=[
		('nshell',c_long),
		('ncxmax',c_long),
		('ncymax',c_long),
		('nczmax',c_long),
		('npoly',c_long),
		('nmono',c_long),
		('internal_poly',c_long),
		('nfil',c_long),
		('nfono', c_long),
		('R_nucleus',c_long),
		('R_nucleusX',c_double),
		('R_nucleusY',c_double),
		('R_nucleusZ',c_double),
		('pswitch',c_long),
		('constvolswitch',c_long),
		('constareaswitch',c_long),
		('stretchswitch',c_long),
		('xkA0',c_double),
		('constvolprecision',c_double),
		('multiprocessing',c_char_p),
		('brezveze0',c_long),
		('brezveze1',c_long),
		('brezveze2',c_long),
		('xk0',c_double),
		('dmax',c_double),
		('dmin_interspecies',c_double),
		('stepsize',c_double),
		('kspring',c_double),
		('xi',c_double),
		('pressure',c_double),
		('iterations',c_long),
		('inititer',c_long),
		('mcsweeps',c_long),
		('quiet',c_long),
		('shc',c_long),
		('number_of_vertice_with_c0',c_long),
		('c0', c_double),
		('w', c_double),
		('F', c_double),
	]
		

class ts_vesicle(Structure):
	_fields_=[
		('vlist', POINTER(ts_vertex_list)),
		('blist', POINTER(ts_bond_list)),
		('tlist', POINTER(ts_triangle_list)),
		('clist', POINTER(ts_cell_list)),
		('nshell', c_uint),
		('bending_rigidity',c_double),
		('dmax',c_double),
		('stepsize',c_double),
		('cm', c_double*3),
		('volume', c_double),
		('sphHarmonics',POINTER(ts_spharm)),
		('poly_list', POINTER(ts_poly_list)),
		('filament_list', POINTER(ts_poly_list)),
		('spring_constant', c_double),
		('pressure', c_double),
		('pswitch', c_int),
		('tape', POINTER(ts_tape)),
		('R_nucleus', c_double),
		('R_nucleusX', c_double),
		('R_nucleusY', c_double),
		('R_nucleusZ', c_double),
		('nucleus_center', c_double *3 ),
		('area', c_double),
	]

ts_cluster._fields_=[('nvtx',c_uint),('idx',c_uint),('vtx', POINTER(POINTER(ts_vertex)))]

class ts_cluster_list(Structure):
	_fields_=[('n',c_uint),('cluster',POINTER(POINTER(ts_cluster)))]




ts=CDLL('libtrisurf.so')



#function call wrappers
def create_vesicle_from_tape(tape):
	"""Using pointer for tape, it creates a vesicle, returning pointer to it."""
	ts.create_vesicle_from_tape.argtypes=POINTER(ts_tape)
	ts.create_vesicle_from_tape.restype=POINTER(ts_vesicle)
	return ts.create_vesicle_from_tape(tape)

def parsetape(filename='tape'):
	"""Loads tape with  filename (if not given it defaults to 'tape'). It returns a pointer to structure for tape"""
	ts.parsetape.restype=POINTER(ts_tape)
	ts.parsetape.argtypes=[c_char_p]
	return ts.parsetape(filename.encode('ascii'))

def parseDump(filename):
	"""Loads a vtu file with 'filename' and creates a vesicle returning pointer to it"""
	ts.parseDump.argtypes=[c_char_p]
	ts.parseDump.restype=POINTER(ts_vesicle)
	vesicle=ts.parseDump(filename.encode('ascii'))
	return vesicle

def single_timestep(vesicle):
	"""Makes a single timestep in simulations. Returns a tuple of vmsrt and bfrt (vertex move success rate and bond flip success rate)"""
	ts.single_timestep.argtypes=[POINTER(ts_vesicle),POINTER(c_double),POINTER(c_double)]
	vmsrt=c_double(0.0)
	bfsrt=c_double(0.0)
	ts.single_timestep(vesicle,byref(vmsrt),byref(bfsrt))
	return (vmsrt.value, bfsrt.value)

def write_vertex_xml_file(vesicle,timestep_no=0):
	"""Writes a vesicle into file with filename 'timestep_XXXXXX.vtu', where XXXXXX is a leading zeroed number given with timestep_no parameter (defaults to 0 if not given"""
	ts.write_vertex_xml_file.argtypess=[POINTER(ts_vesicle),c_int]
	ts.write_vertex_xml_file(vesicle,c_int(timestep_no))


def vesicle_free(vesicle):
	"""Free memory of the whole vesicle"""
	ts.vesicle_free.argtypes=[POINTER(ts_vesicle)]
	ts.vesicle_free(vesicle)

def vesicle_volume(vesicle):
	ts.vesicle_volume.argtypes=[POINTER(ts_vesicle)]
	ts.vesicle_volume(vesicle)

def vesicle_area(vesicle):
	ts.vesicle_area.argtypes=[POINTER(ts_vesicle)]
	ts.vesicle_area(vesicle)

def gyration_eigen(vesicle):
	ts.gyration_eigen.argtypes=[POINTER(ts_vesicle), POINTER(c_double), POINTER(c_double), POINTER(c_double)]
	l1=c_double(0.0)
	l2=c_double(0.0)
	l3=c_double(0.0)
	ts.gyration_eigen(vesicle , byref(l1), byref(l2), byref(l3))
	return (l1.value, l2.value, l3.value)

def vesicle_meancurvature(vesicle):
	ts.vesicle_meancurvature.argtypes=[POINTER(ts_vesicle)]
	ts.vesicle_meancurvature.restype=c_double
	return ts.vesicle_meancurvature(vesicle)

def init_cluster_list():
	ts.init_cluster_list.restype=POINTER(ts_cluster_list)
	ret=ts.init_cluster_list()
	return ret

def clusterize_vesicle(vesicle, cluster_list):
	ts.clusterize_vesicle.argtypes=[POINTER(ts_vesicle), POINTER(ts_cluster_list)]
	ts.clusterize_vesicle(vesicle, cluster_list)

def cluster_list_free(cluster_list):
	"""Free memory of cluster list"""
	ts.cluster_list_free.argtypes=[POINTER(ts_cluster_list)]
	ts.cluster_list_free(cluster_list)

def stretchenergy(vesicle, triangle):
	ts.stretchenergy.argtypes=[POINTER(ts_vesicle), POINTER(ts_triangle)]
	ts.stretchenergy(vesicle,triangle)

def get_absolute_ulm2(vesicle,l,m):
	ts.get_absolute_ulm2.argtypes=[POINTER(ts_vesicle), c_double, c_double]
	ts.get_absolute_ulm2.restype=c_double
	ret=ts.get_absolute_ulm2(vesicle,l,m)
	return ret

def getR0(vesicle):
	ts.getR0.argtypes=[POINTER(ts_vesicle)]
	ts.getR0.restype=c_double
	r0=ts.getR0(vesicle)
	return r0

def preparationSh(vesicle,r0):
	ts.preparationSh.argtypes=[POINTER(ts_vesicle), c_double]
	ts.preparationSh(vesicle,r0)

def calculateUlmComplex(vesicle):
	ts.calculateUlmComplex.argtypes=[POINTER(ts_vesicle)]
	ts.calculateUlmComplex(vesicle)


def Ulm2Complex2String(vesicle):
	ts.Ulm2Complex2String.argtypes=[POINTER(ts_vesicle)]
	ts.Ulm2Complex2String.restype=c_char_p
	string=ts.Ulm2Complex2String(vesicle)
	return string

def freeUlm2String(string):
	ts.freeUlm2String.argtypes=[c_char_p]
	ts.freeUlm2String(string)


#This function seems not to exist!!!
#def solve_for_ulm2(vesicle):
#	ts.solve_for_ulm2.argtypes=[POINTER(ts_vesicle)]
#	ts.solve_for_ulm2(vesicle)

def mean_curvature_and_energy(vesicle):
	ts.mean_curvature_and_energy.argtypes=[POINTER(ts_vesicle)]
	ts.mean_curvature_and_energy(vesicle)	

