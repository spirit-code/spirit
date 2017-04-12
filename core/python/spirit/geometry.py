import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### Imports
from spirit.scalar import scalar
import spirit.system as system

import numpy as np

### Get Bounds
_Get_Bounds          = _spirit.Geometry_Get_Bounds
_Get_Bounds.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Bounds.restype  = None
def Get_Bounds(p_state, min, max, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _min = vec3(*min)
    _max = vec3(*max)
    _Get_Bounds(p_state, _min, _max, idx_image, idx_chain)
    for i in range(3):
        min[i] = _min[i]
        max[i] = _max[i]

### Get Center
_Get_Center          = _spirit.Geometry_Get_Center
_Get_Center.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Center.restype  = None
def Get_Center(p_state, center, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _center = vec3(*center)
    _Get_Center(p_state, _center, idx_image, idx_chain)
    for i in range(3):
        center[i] = _center[i]

### Get Basis vectors
_Get_Basis_Vectors          = _spirit.Geometry_Get_Basis_Vectors
_Get_Basis_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Basis_Vectors.restype  = None
def Get_Basis_Vectors(p_state, a, b, c, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _a = vec3(*a)
    _b = vec3(*b)
    _c = vec3(*c)
    _Get_Basis_Vectors(p_state, _a, _b, _c, idx_image, idx_chain)
    for i in range(3):
        a[i] = _a[i]
        b[i] = _b[i]
        c[i] = _c[i]

### Get N Cells
_Get_N_Cells          = _spirit.Geometry_Get_N_Cells
_Get_N_Cells.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_Get_N_Cells.restype  = None
def Get_N_Cells(p_state, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_int * 3
    n_cells = vec3(*[-1,-1,-1])
    _Get_N_Cells(p_state, n_cells, idx_image, idx_chain)
    return int(n_cells[0]), int(n_cells[1]), int(n_cells[2])

### Get Translation Vectors
_Get_Translation_Vectors          = _spirit.Geometry_Get_Translation_Vectors
_Get_Translation_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Translation_Vectors.restype  = None
def Get_Translation_Vectors(p_state, ta, tb, tc, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Get_Translation_Vectors(p_state, vec3(*ta), vec3(*tb), vec3(*tc), idx_image, idx_chain)

### Get Translation Vectors
_Get_Dimensionality          = _spirit.Geometry_Get_Dimensionality
_Get_Dimensionality.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Dimensionality.restype  = ctypes.c_int
def Get_Dimensionality(p_state, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    return int(_Get_Dimensionality(p_state, idx_image, idx_chain))

### Get Pointer to Spin Positions
_Get_Spin_Positions            = _spirit.Geometry_Get_Spin_Positions
_Get_Spin_Positions.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Spin_Positions.restype    = ctypes.POINTER(scalar)
def Get_Spin_Positions(p_state, idx_image=-1, idx_chain=-1):
    nos = system.Get_NOS(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Spin_Positions(p_state, idx_image, idx_chain)
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = np.frombuffer(array_pointer.contents)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view