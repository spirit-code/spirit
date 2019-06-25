"""
Geometry
====================

Change or get info on the current geometrical configuration, e.g.
number of cells in the three crystal translation directions.
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

### Imports
from spirit.scalar import scalar
from spirit import system

import numpy as np

### Bravais lattice types
BRAVAIS_LATTICE_IRREGULAR   = 0
BRAVAIS_LATTICE_RECTILINEAR = 1
BRAVAIS_LATTICE_SC          = 2
BRAVAIS_LATTICE_HEX2D       = 3
BRAVAIS_LATTICE_HEX2D_60    = 4
BRAVAIS_LATTICE_HEX2D_120   = 5
BRAVAIS_LATTICE_HCP         = 6
BRAVAIS_LATTICE_BCC         = 7
BRAVAIS_LATTICE_FCC         = 8

### ---------------------------------- Set ----------------------------------

_Set_Bravais_Lattice_Type          = _spirit.Geometry_Set_Bravais_Lattice_Type
_Set_Bravais_Lattice_Type.argtypes = [ctypes.c_void_p, ctypes.c_int]
_Set_Bravais_Lattice_Type.restype  = None
def set_bravais_lattice_type(p_state, lattice_type, idx_image=-1, idx_chain=-1):
    """Set the bravais vectors to a pre-defined lattice type:

    - sc:       simple cubic
    - bcc:      body centered cubic
    - fcc:      face centered cubic
    - hex2d:    hexagonal (120deg)
    - hed2d120: hexagonal (120deg)
    - hex2d60:  hexagonal (60deg)
    """
    _Set_Bravais_Lattice_Type(ctypes.c_void_p(p_state), ctypes.c_int(lattice_type))

_Set_N_Cells          = _spirit.Geometry_Set_N_Cells
_Set_N_Cells.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_Set_N_Cells.restype  = None
def set_n_cells(p_state, n_cells=[1, 1, 1], idx_image=-1, idx_chain=-1):
    """Set the number of basis cells along the three bravais vectors."""
    vec3 = ctypes.c_int * 3
    _Set_N_Cells(ctypes.c_void_p(p_state), vec3(*n_cells))

_Set_mu_s             = _spirit.Geometry_Set_mu_s
_Set_mu_s.argtypes    = [ctypes.c_void_p, ctypes.c_float,
                          ctypes.c_int, ctypes.c_int]
_Set_mu_s.restype     = None
def set_mu_s(p_state, mu_s, idx_image=-1, idx_chain=-1):
    """Set the magnetic moment of all atoms."""
    _Set_mu_s(ctypes.c_void_p(p_state), ctypes.c_float(mu_s),
              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_Cell_Atom_Types          = _spirit.Geometry_Set_Cell_Atom_Types
_Set_Cell_Atom_Types.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
_Set_Cell_Atom_Types.restype  = None
def set_cell_atom_types(p_state, atom_types, idx_image=-1, idx_chain=-1):
    """Set the atom types of the atoms in the basis cell."""
    n = len(atom_types)
    vec = ctypes.c_int * n
    _Set_Cell_Atom_Types(ctypes.c_void_p(p_state), ctypes.c_int(n), vec(*atom_types))

_Set_Bravais_Vectors             = _spirit.Geometry_Set_Bravais_Vectors
_Set_Bravais_Vectors.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
_Set_Bravais_Vectors.restype     = None
def set_bravais_vectors(p_state, ta=[1.0, 0.0, 0.0], tb=[0.0, 1.0, 0.0], tc=[0.0, 0.0, 1.0], idx_image=-1, idx_chain=-1):
    """Manually specify the bravais vectors."""
    vec3 = ctypes.c_float * 3
    _Set_Bravais_Vectors(ctypes.c_void_p(p_state), vec3(ta), vec3(tb), vec3(tc))

_Set_Lattice_Constant             = _spirit.Geometry_Set_Lattice_Constant
_Set_Lattice_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_float]
_Set_Lattice_Constant.restype     = None
def set_lattice_constant(p_state, lattice_constant, idx_image=-1, idx_chain=-1):
    """Set the global lattice scaling constant."""
    _Set_Lattice_Constant(p_state, ctypes.c_float(lattice_constant))

### ---------------------------------- Get ----------------------------------

_Get_Bounds          = _spirit.Geometry_Get_Bounds
_Get_Bounds.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Bounds.restype  = None
def get_bounds(p_state, idx_image=-1, idx_chain=-1):
    """Get the bounds of the system in global coordinates.

    Returns two arrays of shape (3) containing minimum and maximum bounds respectively.
    """
    _min = (3*ctypes.c_float)()
    _max = (3*ctypes.c_float)()
    _Get_Bounds(ctypes.c_void_p(p_state), _min, _max,
                ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [_min[i] for i in range(3)], [_max[i] for i in range(3)]

_Get_Center          = _spirit.Geometry_Get_Center
_Get_Center.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Center.restype  = None
def get_center(p_state, idx_image=-1, idx_chain=-1):
    """Get the center of the system in global coordinates.

    Returns an array of shape (3).
    """
    _center = (3*ctypes.c_float)()
    _Get_Center(ctypes.c_void_p(p_state), _center, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [_center[i] for i in range(3)]

_Get_Bravais_Lattice_Type          = _spirit.Geometry_Get_Bravais_Lattice_Type
_Get_Bravais_Lattice_Type.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Bravais_Lattice_Type.restype  = ctypes.c_int
def get_bravais_lattice_type(p_state, idx_image=-1, idx_chain=-1):
    """Get the bravais lattice type corresponding to one of the integers defined above."""
    return int(_Get_Bravais_Lattice_Type(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                 ctypes.c_int(idx_chain)))

_Get_Bravais_Vectors          = _spirit.Geometry_Get_Bravais_Vectors
_Get_Bravais_Vectors.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Get_Bravais_Vectors.restype  = None
def get_bravais_vectors(p_state, idx_image=-1, idx_chain=-1):
    """Get the Bravais vectors.

    Returns three arrays of shape (3).
    """
    _a = (3*ctypes.c_float)()
    _b = (3*ctypes.c_float)()
    _c = (3*ctypes.c_float)()
    _Get_Bravais_Vectors(ctypes.c_void_p(p_state), _a, _b, _c,
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [a for a in _a], [b for b in _b], [c for c in _c]

_Get_N_Cells          = _spirit.Geometry_Get_N_Cells
_Get_N_Cells.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_Get_N_Cells.restype  = None
def get_n_cells(p_state, idx_image=-1, idx_chain=-1):
    """Get the number of basis cells along the three bravais vectors.

    Returns an array of shape (3).
    """
    n_cells = (3*ctypes.c_int)()
    _Get_N_Cells(ctypes.c_void_p(p_state), n_cells, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [n for n in n_cells]

_Get_Dimensionality          = _spirit.Geometry_Get_Dimensionality
_Get_Dimensionality.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Dimensionality.restype  = ctypes.c_int
def get_dimensionality(p_state, idx_image=-1, idx_chain=-1):
    """Get the dimensionality of the geometry."""
    return int(_Get_Dimensionality(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get Pointer to Spin Positions
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Positions            = _spirit.Geometry_Get_Positions
_Get_Positions.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Positions.restype    = ctypes.POINTER(scalar)
def get_positions(p_state, idx_image=-1, idx_chain=-1):
    """Returns a `numpy.array_view` of shape (NOS, 3) with the components of each spins position.

    Changing the contents of this array_view will have direct effect on the state and should not be done.
    """
    nos = system.get_nos(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Positions(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = np.frombuffer(array_pointer.contents, dtype=scalar)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view

_Get_N_Cell_Atoms          = _spirit.Geometry_Get_N_Cell_Atoms
_Get_N_Cell_Atoms.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_N_Cell_Atoms.restype  = ctypes.c_int
def get_n_cell_atoms(p_state, idx_image=-1, idx_chain=-1):
    """Get the number of atoms in the basis cell."""
    return int(_Get_N_Cell_Atoms(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get Pointer to atom types
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Atom_Types            = _spirit.Geometry_Get_Atom_Types
_Get_Atom_Types.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Atom_Types.restype    = ctypes.POINTER(ctypes.c_int)
def get_atom_types(p_state, idx_image=-1, idx_chain=-1):
    """Get the types of all atoms as a `numpy.array_view` of shape (NOS).

    If e.g. disorder is activated, this allows to view and manipulate the types of individual atoms.
    """
    nos = system.get_nos(p_state, idx_image, idx_chain)
    ArrayType = ctypes.c_int*nos
    Data = _Get_Atom_Types(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = np.frombuffer(array_pointer.contents, dtype=ctypes.c_int)
    array_view = array.view()
    return array_view