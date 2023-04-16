#!/usr/bin/env python
# coding: utf-8
"""Wrapper for the trisurf (new cluster_trisurf version)

Has python binding to the library using ctypes.CDLL and classes for the important structres (ts_vertex, ts_tape, ts_vesicle)
"""

########################
# 0: imports and paths #
########################

from ctypes import *
import os

# raise DeprecationWarning("Wrapper has not been updated to the new versions of trisurf")
path_to_trisurf_library =  '/opt/workspace/msc_project/cluster-trisurf/src/.libs/libtrisurf.so'

if not os.path.isfile(path_to_trisurf_library):
	print("Library not found in {path_to_trisurf_library=}. Please update the wrapper with the right location!")

#########################################
# 1: constants, type aliases, and enums #
#########################################

TS_SUCCESS=0
TS_FAIL=1

TS_ID_FILAMENT=1

TS_COORD_CARTESIAN=0
TS_COORD_SPHERICAL=1
TS_COORD_CYLINDRICAL=2

# type aliases
ts_massive_idx = c_long
ts_cell_idx = c_uint
ts_idx = c_int
ts_small_idx = c_byte
ts_flag = c_char
ts_bool = c_char

# enums vertex_type 
is_bonding_vtx=1     # bonding vertex, form bond with other bonding vertices
is_active_vtx=2      # active vertex under normally directed force
is_adhesive_vtx=4    # adhesive vertex, subject to adhesion energy near surfaces
is_anisotropic_vtx=8 # anisotropic vertex, requires computing full curvature characteristic
is_reserved_0_vtx=16 # reserved type
is_vicsek_vtx=32     # vertex under vicsek neighbor-influenced force
is_edge_vtx=64       # edge vertex has unordered tristars
is_ghost_vtx=-128    # ghost vertex can only be moved artificially
 
# enum bond_model_type
is_bonding_type_specific=1
is_anisotropic_bonding_nematic=2

# enum curvature_model_type
to_disable_calculate_laplace_beltrami=64  # The original isotropic mean curvature calculation
to_calculate_sum_angle=1             # "isotropic" gaussian curvature calculation
to_calculate_shape_operator=2        # new anisotropic calculation. if to_use is not enabled, this is saved but not used!
to_update_director_shapeless=4       # update director without shape operator
to_use_shape_operator_energy=8       # actually use the new energy, rather than just calculate and save.
to_use_shape_for_anisotropy_only=16  # use the shape method, but only for anisotropic vertices
to_not_rotate_directors=32           # do not rotate directro as a monte carlo step
to_use_sum_angle_for_kx2_only=128    # only calculate the sum angle formula for vtx with kx2!=0

model_laplace_beltrami_only=0        # the original method
model_isotropic=1                    # calculate gaussian curvature energy
model_shape_operator_only=74         # use shape operator only
model_debug_old_energy=7             # calculate everything but use old energy
model_debug_new_energy=15            # calculate everything but use new shape operator energy
model_debug_parallerl_transport_directors=35 # prevent director random move
model_debug_assumed_final=146        # anisotropic shape method only for anisotropic vertices, old method otherwise
    

# enum force_model_type
model_vertex_meanW=0           # regular bonding model
model_active_neigh_interfer=1  # force is proportional to # non-active neighbors
model_concave_neigh_interfer=2 # force is proportional to # non-concave neighbors
model_concave_neigh_disable=3  # force is disabled by any concave neighbor
is_vicsek_model=16             # use a vicsek model. All vicsek model should have (model_vicsek_x&is_vicsek_model) true
model_vicsek_1overR=17         # use vicsek model with 1/R weights

# enum adhesion_type
adhesion_step_potential=1
adhesion_parabolic_potential=2

# enum adhesion_geometry_type
model_plane_potential=1
model_spherical_potential=2
model_cylindrical_potential=3



###################################
# 2: structures of the simulation #
###################################

class ts_coord(Structure):
	_fields_=[
		("e1", c_double),
		("e2", c_double),
		("e3", c_double),
		("coord_type", c_uint)
		]
	
# need to be defined before the fields??
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
	("x",c_double),
	("y",c_double),
	("z",c_double),
	("mean_curvature",c_double),
	("gaussian_curvature",c_double),
	("mean_energy",c_double),
	("gaussian_energy",c_double),
	("energy",c_double),
	("xx",c_double),
	("xx2",c_double),
	("w",c_double),
	("c",c_double),
	("nx",c_double),
	("ny",c_double),
	("nz",c_double),
	("nx2",c_double),
	("ny2",c_double),
	("nz2",c_double),
	("f",c_double),
	("fx",c_double),
	("fy",c_double),
	("fz",c_double),
	("ad_w",c_double),
	("d",c_double),
	("dx",c_double),
	("dy",c_double),
	("dz",c_double),
	("eig0",c_double*3),
	("eig1",c_double*3),
	("eig2",c_double*3),
	("new_c1",c_double),
	("new_c2",c_double),
	("eig_v0",c_double),
	("eig_v1",c_double),
	("eig_v2",c_double),
	("mean_curvature2",c_double),
	("gaussian_curvature2",c_double),
	("mean_energy2",c_double),
	("gaussian_energy2",c_double),
	("neigh", POINTER(POINTER(ts_vertex))),
	("tristar",POINTER(POINTER(ts_triangle))),
	("bond",POINTER(POINTER(ts_bond))),
	("cell",POINTER(POINTER(ts_cell))),
	("grafted_poly",POINTER(POINTER(ts_poly))),
	('cluster',POINTER(ts_cluster)),
	("idx",ts_idx),
	("id",ts_idx),
	("neigh_no",ts_small_idx),
	("tristar_no",ts_small_idx),
	("bond_no",ts_small_idx),
	("type",ts_flag)
]

class ts_vertex_list(Structure):
	_fields_=[('vtx',POINTER(POINTER(ts_vertex))),('n',ts_idx),]

ts_bond._fields_=[
		('bond_length',c_double),
		('energy',c_double),
		('x',c_double),
		('y',c_double),
		('z',c_double),
		('vtx1', POINTER(ts_vertex)),
		('vtx2', POINTER(ts_vertex)),
		('idx',ts_idx),
	]
class ts_bond_list(Structure):
	_fields_=[('bond',POINTER(POINTER(ts_bond))),('n', ts_idx),]

ts_triangle._fields_=[
		('xnorm', c_double),
		('ynorm', c_double),
		('znorm', c_double),
		('xcirc', c_double),
		('ycirc', c_double),
		('zcirc', c_double),
		('area', c_double),
		('volume', c_double),
		('energy', c_double),
		('vertex', POINTER(ts_vertex)*3),
		('neigh', POINTER(POINTER(ts_triangle))),
		('idx',ts_idx),
		('neigh_no',ts_small_idx),
	]

class ts_triangle_list(Structure):
	_fields_=[('a0',c_double),('tria', POINTER(POINTER(ts_triangle))),('n',ts_idx),]


ts_cell._fields_=[
	('vertex', POINTER(POINTER(ts_vertex))),
	('idx', ts_cell_idx),
	('nvertex', ts_small_idx),
	]		

class ts_cell_list(Structure):
	_fields_=[
		("dcell",c_double),
		('shift', c_double*3),
		('dmin_interspecies', c_double),
		('cell',POINTER(POINTER(ts_cell))),
		('ncmax', ts_cell_idx*3),
		('cellno', ts_cell_idx),
		('max_occupancy', ts_small_idx),
	]

class ts_spharm(Structure):
	_fields_=[
		("vtx_relR", POINTER(c_double)),
		("vtx_solAngle", POINTER(c_double)),
		('ulm', POINTER(POINTER(c_double))),
		('co',POINTER(POINTER(c_double))),
		('Ylmi', POINTER(POINTER(POINTER(c_double)))),
		('sumUlm2', POINTER(POINTER(c_double))),
		('ulmComplex', POINTER(POINTER(c_double))), # temporary solution (?) 
		('n_vtx',ts_idx),
		('l',c_uint),
		('N', c_uint),
		]

ts_poly._fields_=[
		('k', c_double),
		('vlist', POINTER(ts_vertex_list)),
		('blist', POINTER(ts_bond_list)),
		('grafted_vtx',POINTER(ts_vertex)),
	]

class ts_poly_list(Structure):
	_fields_=[('poly',POINTER(POINTER(ts_poly))),('n',ts_idx),]

class ts_confinment_plane(Structure):
	_field_=[("z_max",c_double),("z_min",c_double),("force_switch",ts_bool)]

class ts_tape(Structure):
	_fields_=[
		("R_nucleus",c_double),
		("R_nucleusX",c_double),
		("R_nucleusY",c_double),
		("R_nucleusZ",c_double),
		("kxA0",c_double),
		("kxV0",c_double),
		("V0",c_double),
		("A0",c_double),
		("Vfraction",c_double),
		("constvolprecision",c_double),
		("xk0",c_double),
		("xk2",c_double),
		("dmax",c_double),
		("dmin_interspecies",c_double),
		("stepsize",c_double),
		("kspring",c_double),
		("xi",c_double),
		("pressure",c_double),
		("c0",c_double),
		("w",c_double),
		("F",c_double),
		("plane_d",c_double),
		("plane_F",c_double),
		("vicsek_strength",c_double),
		("vicsek_strength",c_double),
		("z_adhesion",c_double),
		("adhesion_radius",c_double),
		("min_dihedral_angle_cosine",c_double),
		("d0",c_double),
		("mcsweeps",ts_massive_idx),
		("random_seed",c_ulong),
		("iterations",ts_idx),
		("inititer",ts_idx),
		("number_of_vertices_with_c0",ts_idx),
		("nshell",c_uint),
		("ncxmax",c_uint),
		("ncymax",c_uint),
		("nczmax",c_uint),
		("npoly",ts_idx),
		("nmono",ts_idx),
		("internal_poly",ts_idx),
		("nfil",ts_idx),
		("nfono",ts_idx),
		("shc",c_uint),
		("pressure_switch",ts_bool),
		("volume_switch",ts_bool),
		("area_switch",ts_bool),
		("quiet",ts_bool),
		("plane_confinment_switch",ts_bool),
		("allow_center_mass_movement",ts_bool),
		("force_balance_along_z_axis",ts_bool),
		("adhesion_geometry",ts_flag),
		("adhesion_model",ts_flag),
		("type_of_bond_model",ts_flag),
		("type_of_curvature_model",ts_flag),
		("type_of_force_model",ts_flag),
	]

		

class ts_vesicle(Structure):
	_fields_=[
		('dmax',c_double),
		('stepsize',c_double),
		('cm', c_double*3),
		("fx",c_double),
		("fy",c_double),
		("fz",c_double),
		('volume', c_double),
		('area', c_double),
		('spring_constant', c_double),
		('pressure', c_double),
		('R_nucleus', c_double),
		('R_nucleusX', c_double),
		('R_nucleusY', c_double),
		('R_nucleusZ', c_double),
		('nucleus_center', c_double *3 ),
		('tape', POINTER(ts_tape)),	
		('sphHarmonics',POINTER(ts_spharm)),
		('poly_list', POINTER(ts_poly_list)),
		('filament_list', POINTER(ts_poly_list)),
		('vlist', POINTER(ts_vertex_list)),
		('blist', POINTER(ts_bond_list)),
		('tlist', POINTER(ts_triangle_list)),
		('clist', POINTER(ts_cell_list)),
		("confinement_plane",ts_confinment_plane),
		('nshell', c_int),
	]

ts_cluster._fields_=[('vtx', POINTER(POINTER(ts_vertex))),('nvtx',ts_idx),('idx',ts_idx)]

class ts_cluster_list(Structure):
	_fields_=[('cluster',POINTER(POINTER(ts_cluster))),('n',ts_idx),]



###############################
# 3: load the trisurf library #
###############################

ts=CDLL(path_to_trisurf_library)


########################
# 4: function wrappers #
########################


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
	ts.write_vertex_xml_file.argtypess=[POINTER(ts_vesicle),ts_idx,POINTER(ts_cluster_list)]
	ts.write_vertex_xml_file(vesicle,ts_idx(timestep_no),POINTER(ts_cluster_list)())


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

def get_absolute_ulm2(vesicle,l,m): # cant find this function anywhere!? maybe it's calculateKc?
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

def direct_force_energy(vesicle, vtx, vtx_old):
		"""update the forces on a vertex and return the work done"""
		ts.direct_force_energy.argtypes=[POINTER(ts_vesicle),POINTER(ts_vertex),POINTER(ts_vertex)]
		ts.direct_force_energy.restype=c_double
		return ts.direct_force_energy(vesicle,vtx,vtx_old)

def update_vertex_force(vesicle, vtx):
		"""update the forces on a vertex using direct_force_energy"""
		direct_force_energy(vesicle,vtx,vtx)
		return vtx.fx, vtx.fy, vtx.fz

def adhesion_energy_diff(vesicle,vtx,vtx_old):
	"""Adhesion energy difference between old and new vertex"""
	ts.adhesion_energy_diff.argtypes=[POINTER(ts_vesicle),POINTER(ts_vertex),POINTER(ts_vertex)]
	ts.adhesion_energy_diff.restype=c_double
	return ts.adhesion_energy_diff(vesicle,vtx,vtx_old)

def adhesion_geometry_distance(vesicle,vtx):
	"""Distance between vertex and the adhesion geometry"""
	ts.adhesion_geometry_distance.argtypes=[POINTER(ts_vesicle),POINTER(ts_vertex)]
	ts.adhesion_geometry_distance.restype=c_double
	return ts.adhesion_geometry_distance(vesicle,vtx)

def adhesion_geometry_side(vesicle,vtx):
	"""is the vertex normal oriented towards the adhesion geometry"""
	ts.adhesion_geometry_side.argtypes=[POINTER(ts_vesicle),POINTER(ts_vertex)]
	ts.adhesion_geometry_side.restype=ts_bool
	return ts.adhesion_geometry_side(vesicle,vtx)

def vertex_adhesion(vesicle,vtx):
	"""energy of a single vertex: piece of adhesion_energy_diff"""
	delta = adhesion_geometry_distance(vesicle,vtx)
	is_oriented = adhesion_geometry_side(vesicle,vtx)
	model = vesicle.tape.adhesion_model
	dz = vesicle.tape.adhesion_cutoff
	if vtx.type&is_adhesive_vtx and is_oriented and delta<=dz:
		if model==adhesion_step_potential:
			return vtx.ad_w
		elif model==adhesion_parabolic_potential:
			return vtx.ad_w*(1-(delta/dz)**2)
	return 0


if __name__=="__main__":
    print(f"running wrapper: {ts.__name__}")