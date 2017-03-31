import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

from spirit.scalar import scalar
import numpy as np

### Get Chain index
_Get_Index          = _spirit.System_Get_Index
_Get_Index.argtypes = [ctypes.c_void_p]
_Get_Index.restype  = ctypes.c_int
def Get_Index(p_state):
    return int(_Get_Index(p_state))


### Get Chain number of images
_Get_NOS            = _spirit.System_Get_NOS
_Get_NOS.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_NOS.restype    = ctypes.c_int
def Get_NOS(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_NOS(p_state, idx_image, idx_chain))

### Get Pointer to Spin Directions
_Get_Spin_Directions            = _spirit.System_Get_Spin_Directions
_Get_Spin_Directions.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Spin_Directions.restype    = ctypes.POINTER(scalar)
def Get_Spin_Directions(p_state, idx_image=-1, idx_chain=-1):
    nos = Get_NOS(p_state, idx_image, idx_chain)
    ArrayType = scalar*3*nos
    Data = _Get_Spin_Directions(p_state, idx_image, idx_chain)
    array_pointer = ctypes.cast(Data, ctypes.POINTER(ArrayType))
    array = np.frombuffer(array_pointer.contents)
    array_view = array.view()
    array_view.shape = (nos, 3)
    return array_view

### Get total Energy
_Get_Energy          = _spirit.System_Get_Energy
_Get_Energy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Energy.restype  = ctypes.c_float
def Get_Energy(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_Energy(p_state, idx_image, idx_chain))

### Get Energy array
_Get_Energy_Array          = _spirit.System_Get_Energy_Array
_Get_Energy_Array.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Energy_Array.restype  = None
def Get_Energy_Array(p_state, energies, idx_image=-1, idx_chain=-1):
    _Get_Energy_Array(p_state, energies, idx_image, idx_chain)

### Get Chain number of images
_Update_Data            = _spirit.System_Update_Data
_Update_Data.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Update_Data.restype    = None
def Update_Data(p_state, idx_image=-1, idx_chain=-1):
    _Update_Data(p_state, idx_image, idx_chain)

### Print Energy array
_Print_Energy_Array            = _spirit.System_Print_Energy_Array
_Print_Energy_Array.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Print_Energy_Array.restype    = None
def Print_Energy_Array(p_state, idx_image=-1, idx_chain=-1):
    _Print_Energy_Array(p_state, idx_image, idx_chain)