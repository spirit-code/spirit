import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Get Chain index
_Get_Index          = _core.System_Get_Index
_Get_Index.argtypes = [ctypes.c_void_p]
_Get_Index.restype  = ctypes.c_int
def Get_Index(p_state):
    return int(_Get_Index(p_state))


### Get Chain number of images
_Get_NOS            = _core.System_Get_NOS
_Get_NOS.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_NOS.restype    = ctypes.c_int
def Get_NOS(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_NOS(p_state, idx_image, idx_chain))

### Get Pointer to Spin Directions
_Get_Spin_Directions            = _core.System_Get_Spin_Directions
_Get_Spin_Directions.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Spin_Directions.restype    = ctypes.POINTER(ctypes.c_float)
def Get_Spin_Directions(p_state, idx_image=-1, idx_chain=-1):
    return ctypes.POINTER(ctypes.c_float)(_Get_Spin_Directions(p_state, idx_image, idx_chain))

### Get total Energy
_Get_Energy          = _core.System_Get_Energy
_Get_Energy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Energy.restype  = ctypes.c_float
def Get_Energy(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_Energy(p_state, idx_image, idx_chain))

### Get Energy array
_Get_Energy_Array          = _core.System_Get_Energy_Array
_Get_Energy_Array.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Energy_Array.restype  = None
def Get_Energy_Array(p_state, energies, idx_image=-1, idx_chain=-1):
    _Get_Energy_Array(p_state, energies, idx_image, idx_chain)

### Get Chain number of images
_Update_Data            = _core.System_Update_Data
_Update_Data.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Update_Data.restype    = None
def Update_Data(p_state, idx_image=-1, idx_chain=-1):
    _Update_Data(p_state, idx_image, idx_chain)