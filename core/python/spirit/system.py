"""
System
====================
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

from spirit.scalar import scalar
from spirit import parameters
from numpy import frombuffer, ndarray as np

### Get Chain index
_Get_Index          = _spirit.System_Get_Index
_Get_Index.argtypes = [ctypes.c_void_p]
_Get_Index.restype  = ctypes.c_int
def get_index(p_state):
    """Returns the index of the currently active image."""
    return int(_Get_Index(ctypes.c_void_p(p_state)))

### Get Chain number of images
_Get_NOS            = _spirit.System_Get_NOS
_Get_NOS.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_NOS.restype    = ctypes.c_int
def get_nos(p_state, idx_image=-1, idx_chain=-1):
    """Returns the number of spins (NOS)."""
    return int(_Get_NOS(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get Pointer to Spin Directions
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Spin_Directions            = _spirit.System_Get_Spin_Directions
_Get_Spin_Directions.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Spin_Directions.restype    = ctypes.POINTER(scalar)
def get_spin_directions(p_state, idx_image=-1, idx_chain=-1):
    """Returns an `numpy.array_view` of shape (NOS, 3) with the components of each spins orientation vector.

    Changing the contents of this array_view will have direct effect on calculations etc.
    """
    nos = get_nos(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Spin_Directions(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = frombuffer(array_pointer.contents, dtype=scalar)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view

### Get Pointer to Effective Field
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Effective_Field            = _spirit.System_Get_Effective_Field
_Get_Effective_Field.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Effective_Field.restype    = ctypes.POINTER(scalar)
def get_effective_field(p_state, idx_image=-1, idx_chain=-1):
    nos = get_nos(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Effective_Field(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = frombuffer(array_pointer.contents, dtype=scalar)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view

### Get Pointer to an eigenmode
# NOTE: Changing the values of the array_view one can alter the value of the data of the state
_Get_Eigenmode            = _spirit.System_Get_Eigenmode
_Get_Eigenmode.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Get_Eigenmode.restype    = ctypes.POINTER(scalar)
def get_eigenmode(p_state, idx_mode, idx_image=-1, idx_chain=-1):
    nos = get_nos(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Eigenmode(ctypes.c_void_p(p_state), ctypes.c_int(idx_mode),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = frombuffer(array_pointer.contents, dtype=scalar)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view

### Get total Energy
_Get_Energy          = _spirit.System_Get_Energy
_Get_Energy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Energy.restype  = ctypes.c_float
def get_energy(p_state, idx_image=-1, idx_chain=-1):
    """Calculates and returns the energy of the system."""
    return float(_Get_Energy(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                             ctypes.c_int(idx_chain)))

### Get Energy
_Get_Eigenvalues          = _spirit.System_Get_Eigenvalues
_Get_Eigenvalues.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Eigenvalues.restype  = None
def get_eigenvalues(p_state, idx_image=-1, idx_chain=-1):
    noe = parameters.ema.getNModes(p_state, idx_image, idx_chain)
    eigenvalues = (noe*ctypes.c_float)()
    _Get_Eigenvalues(ctypes.c_void_p(p_state), eigenvalues, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return eigenvalues

# NOTE: excluded since there is no clean way to get the C++ pairs
### Get Energy array
# _Get_Energy_Array          = _spirit.System_Get_Energy_Array
# _Get_Energy_Array.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
#                               ctypes.c_int, ctypes.c_int]
# _Get_Energy_Array.restype  = None
# def Get_Energy_Array(p_state, idx_image=-1, idx_chain=-1):
#     Energies
#     _Get_Energy_Array(ctypes.c_void_p(p_state), energies,
#                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Get Chain number of images
_Update_Data            = _spirit.System_Update_Data
_Update_Data.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Update_Data.restype    = None
def update_data(p_state, idx_image=-1, idx_chain=-1):
    """TODO: document when this needs to be called."""
    _Update_Data(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### Eigenmodes
_Eigenmodes          = _spirit.System_Update_Eigenmodes
_Eigenmodes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Eigenmodes.restype  = None
def update_eigenmodes(p_state, idx_image=-1, idx_chain=-1):
    """Calculates eigenmodes of a system according to EMA parameters.
    This needs to be called or eigenmodes need to be read in before they can be used by other functions
    (e.g. writing them to a file).
    """
    spiritlib.wrap_function(_Eigenmodes, [ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                         ctypes.c_int(idx_chain)])

### Print Energy array
_Print_Energy_Array            = _spirit.System_Print_Energy_Array
_Print_Energy_Array.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Print_Energy_Array.restype    = None
def print_energy_array(p_state, idx_image=-1, idx_chain=-1):
    """Print the energy array of the state to the console."""
    _Print_Energy_Array(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))