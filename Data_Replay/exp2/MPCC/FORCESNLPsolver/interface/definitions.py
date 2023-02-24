import numpy
import ctypes

name = "FORCESNLPsolver"
requires_callback = True
lib = "lib/libFORCESNLPsolver.so"
lib_static = "lib/libFORCESNLPsolver.a"
c_header = "include/FORCESNLPsolver.h"
nstages = 30

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  5,   1),    5),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (270,   1),  270),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (3510,   1), 3510)]

# Output                | Type    | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x02"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x03"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x04"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x05"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x06"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x07"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x08"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x09"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x10"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x11"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x12"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x13"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x14"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x15"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x16"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x17"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x18"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x19"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x20"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x21"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x22"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x23"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x24"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x25"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x26"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x27"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x28"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x29"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9),
 ("x30"                 , ""               , ctypes.c_double, numpy.float64,     (  9,),    9)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
 ('it2opt', ctypes.c_int),
 ('res_eq', ctypes.c_double),
 ('res_ineq', ctypes.c_double),
 ('rsnorm', ctypes.c_double),
 ('rcompnorm', ctypes.c_double),
 ('pobj', ctypes.c_double),
 ('dobj', ctypes.c_double),
 ('dgap', ctypes.c_double),
 ('rdgap', ctypes.c_double),
 ('mu', ctypes.c_double),
 ('mu_aff', ctypes.c_double),
 ('sigma', ctypes.c_double),
 ('lsit_aff', ctypes.c_int),
 ('lsit_cc', ctypes.c_int),
 ('step_aff', ctypes.c_double),
 ('step_cc', ctypes.c_double),
 ('solvetime', ctypes.c_double),
 ('fevalstime', ctypes.c_double),
 ('solver_id', ctypes.c_int * 8)
]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1), 
	(9, 5, 25, 117, 5, 5, 25, 1)
]