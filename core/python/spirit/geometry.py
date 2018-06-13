import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### Imports
from spirit.scalar import scalar
from spirit import system

import numpy as np

### ---------------------------------- Set ----------------------------------

### Set the type of Bravais lattice. Can be e.g. "sc" or "bcc"
_Set_Bravais_Lattice          = _spirit.Geometry_Set_Bravais_Lattice
_Set_Bravais_Lattice.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_Set_Bravais_Lattice.restype  = None
def setBravaisLattice(p_state, lattice, idx_image=-1, idx_chain=-1):
    _Set_Bravais_Lattice(ctypes.c_void_p(p_state), ctypes.c_char_p(lattice.encode('utf-8')))

### Set number of cells in bravais lattice directions a, b, c
_Set_N_Cells          = _spirit.Geometry_Set_N_Cells
_Set_N_Cells.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_Set_N_Cells.restype  = None
def setNCells(p_state, n_cells=[1, 1, 1], idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_int * 3
    _Set_N_Cells(ctypes.c_void_p(p_state), vec3(*n_cells))

### Set the types of the atoms in a basis cell
_Set_Cell_Atom_Types          = _spirit.Geometry_Set_Cell_Atom_Types
_Set_Cell_Atom_Types.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
_Set_Cell_Atom_Types.restype  = None
def setCellAtomTypes(p_state, atom_types, idx_image=-1, idx_chain=-1):
    n = len(atom_types)
    vec = ctypes.c_int * n
    _Set_Cell_Atom_Types(ctypes.c_void_p(p_state), ctypes.c_int(n), vec(*atom_types))

### Set the bravais vectors
_Set_Bravais_Vectors             = _spirit.Geometry_Set_Bravais_Vectors
_Set_Bravais_Vectors.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_Set_Bravais_Vectors.restype     = None
def setBravaisVectors(p_state, ta=[1.0, 0.0, 0.0], tb=[0.0, 1.0, 0.0], tc=[0.0, 0.0, 1.0], idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Set_Bravais_Vectors(ctypes.c_void_p(p_state), vec3(ta), vec3(tb), vec3(tc))

### Set the overall lattice constant
_Set_Lattice_Constant             = _spirit.Geometry_Set_Lattice_Constant
_Set_Lattice_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_float]
_Set_Lattice_Constant.restype     = None
def setLatticeConstant(p_state, lattice_constant, idx_image=-1, idx_chain=-1):
    _Set_Lattice_Constant(p_state, ctypes.c_float(lattice_constant))

### ---------------------------------- Get ----------------------------------

### Get Bounds
_Get_Bounds          = _spirit.Geometry_Get_Bounds
_Get_Bounds.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Bounds.restype  = None
def Get_Bounds(p_state, idx_image=-1, idx_chain=-1):
    _min = (3*ctypes.c_float)()
    _max = (3*ctypes.c_float)()
    _Get_Bounds(ctypes.c_void_p(p_state), _min, _max, 
                ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [_min[i] for i in range(3)], [_max[i] for i in range(3)] 

### Get Center
_Get_Center          = _spirit.Geometry_Get_Center
_Get_Center.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Center.restype  = None
def Get_Center(p_state, idx_image=-1, idx_chain=-1):
    _center = (3*ctypes.c_float)()
    _Get_Center(ctypes.c_void_p(p_state), _center, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [_center[i] for i in range(3)]

### Get Bravais lattice type
_Get_Bravais_Type          = _spirit.Geometry_Get_Bravais_Type
_Get_Bravais_Type.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Bravais_Type.restype  = ctypes.c_int
def Get_Bravais_Type(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_Bravais_Type(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                 ctypes.c_int(idx_chain)))

### Get Bravais vectors
_Get_Bravais_Vectors          = _spirit.Geometry_Get_Bravais_Vectors
_Get_Bravais_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                               ctypes.c_int, ctypes.c_int]
_Get_Bravais_Vectors.restype  = None
def Get_Bravais_Vectors(p_state, idx_image=-1, idx_chain=-1):
    _a = (3*ctypes.c_float)()
    _b = (3*ctypes.c_float)()
    _c = (3*ctypes.c_float)()
    _Get_Bravais_Vectors(ctypes.c_void_p(p_state), _a, _b, _c, 
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [a for a in _a], [b for b in _b], [c for c in _c]
    
### Get N Cells
_Get_N_Cells          = _spirit.Geometry_Get_N_Cells
_Get_N_Cells.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_Get_N_Cells.restype  = None
def Get_N_Cells(p_state, idx_image=-1, idx_chain=-1):
    n_cells = (3*ctypes.c_int)()
    _Get_N_Cells(ctypes.c_void_p(p_state), n_cells, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [n for n in n_cells]

### Get Translation Vectors
_Get_Translation_Vectors          = _spirit.Geometry_Get_Translation_Vectors
_Get_Translation_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                                     ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                     ctypes.c_int, ctypes.c_int]
_Get_Translation_Vectors.restype  = None
def Get_Translation_Vectors(p_state, idx_image=-1, idx_chain=-1):
    ta = (3*ctypes.c_float)()
    tb = (3*ctypes.c_float)()
    tc = (3*ctypes.c_float)()
    _Get_Translation_Vectors(ctypes.c_void_p(p_state), ta, tb, tc, 
                             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [a for a in ta], [b for b in tb], [c for c in tc]

### Get Translation Vectors
_Get_Dimensionality          = _spirit.Geometry_Get_Dimensionality
_Get_Dimensionality.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Dimensionality.restype  = ctypes.c_int
def Get_Dimensionality(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_Dimensionality(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                   ctypes.c_int(idx_chain)))

### Get Pointer to Spin Positions
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Positions            = _spirit.Geometry_Get_Positions
_Get_Positions.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Positions.restype    = ctypes.POINTER(scalar)
def Get_Positions(p_state, idx_image=-1, idx_chain=-1):
    nos = system.Get_NOS(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Positions(ctypes.c_void_p(p_state), 
                               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = np.frombuffer(array_pointer.contents, dtype=scalar)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view

### Get Pointer to atom types
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Atom_Types            = _spirit.Geometry_Get_Atom_Types
_Get_Atom_Types.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Atom_Types.restype    = ctypes.POINTER(ctypes.c_int)
def Get_Atom_Types(p_state, idx_image=-1, idx_chain=-1):
    nos = system.Get_NOS(p_state, idx_image, idx_chain)
    ArrayType = ctypes.c_int*nos
    Data = _Get_Atom_Types(ctypes.c_void_p(p_state), 
                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = np.frombuffer(array_pointer.contents, dtype=ctypes.c_int)
    array_view = array.view()
    return array_view