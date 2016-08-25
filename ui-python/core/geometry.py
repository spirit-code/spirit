import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Get Bounds
_Get_Bounds          = _core.Geometry_Get_Bounds
_Get_Bounds.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_Get_Bounds.restype  = None
def Get_Bounds(p_state, min, max):
    vec3 = ctypes.c_double * 3
    _Get_Bounds(p_state, vec3(*min), vec3(*max))

### Get Center
_Get_Center          = _core.Geometry_Get_Center
_Get_Center.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
_Get_Center.restype  = None
def Get_Center(p_state, center):
    vec3 = ctypes.c_double * 3
    _Get_Center(p_state, vec3(*center))

### Get Basis vectors
_Get_Basis_Vectors          = _core.Geometry_Get_Basis_Vectors
_Get_Basis_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_Get_Basis_Vectors.restype  = None
def Get_Basis_Vectors(p_state, a, b, c):
    vec3 = ctypes.c_double * 3
    _Get_Basis_Vectors(p_state, vec3(*a), vec3(*b), vec3(*c))

### Get N Cells
_Get_N_Cells          = _core.Geometry_Get_N_Cells
_Get_N_Cells.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_Get_N_Cells.restype  = None
def Get_N_Cells(p_state, na, nb, nc):
    _Get_N_Cells(p_state, na, nb, nc)

### Get Translation Vectors
_Get_Translation_Vectors          = _core.Geometry_Get_Translation_Vectors
_Get_Translation_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
_Get_Translation_Vectors.restype  = None
def Get_Translation_Vectors(p_state, ta, tb, tc):
    vec3 = ctypes.c_double * 3
    _Get_Translation_Vectors(p_state, vec3(*ta), vec3(*tb), vec3(*tc))

### Get Translation Vectors
_Is_2D          = _core.Geometry_Is_2D
_Is_2D.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_Is_2D.restype  = None
def Is_2D(p_state):
    vec3 = ctypes.c_double * 3
    return ctypes.c_bool(_Is_2D(p_state))