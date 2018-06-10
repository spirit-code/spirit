import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

## ---------------------------------- Set ----------------------------------

### Set number of modes
_Set_EMA_N_Modes          = _spirit.Parameters_Set_EMA_N_Modes
_Set_EMA_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
_Set_EMA_N_Modes.restype  = None
def setNModes(p_state, n_modes, idx_image=-1, idx_chain=-1):
    _Set_EMA_N_Modes(ctypes.c_void_p(p_state), ctypes.c_int(n_modes),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set index of mode to follow
_Set_EMA_N_Mode_Follow          = _spirit.Parameters_Set_EMA_N_Mode_Follow
_Set_EMA_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
_Set_EMA_N_Mode_Follow.restype  = None
def setNModeFollow(p_state, n_mode, idx_image=-1, idx_chain=-1):
    _Set_EMA_N_Mode_Follow(ctypes.c_void_p(p_state), ctypes.c_int(n_mode),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

## ---------------------------------- Get ----------------------------------

### Get number of modes
_Get_EMA_N_Modes          = _spirit.Parameters_Get_EMA_N_Modes
_Get_EMA_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_EMA_N_Modes.restype  = ctypes.c_int
def getNModes(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_EMA_N_Modes(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get index of mode to follow
_Get_EMA_N_Mode_Follow          = _spirit.Parameters_Get_EMA_N_Mode_Follow
_Get_EMA_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_EMA_N_Mode_Follow.restype  = ctypes.c_int
def getNModeFollow(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_EMA_N_Mode_Follow(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))