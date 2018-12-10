import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

## ---------------------------------- Set ----------------------------------

### Set number of modes
_EMA_Set_N_Modes          = _spirit.Parameters_EMA_Set_N_Modes
_EMA_Set_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
_EMA_Set_N_Modes.restype  = None
def set_n_modes(p_state, n_modes, idx_image=-1, idx_chain=-1):
    _EMA_Set_N_Modes(ctypes.c_void_p(p_state), ctypes.c_int(n_modes),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set index of mode to follow
_EMA_Set_N_Mode_Follow          = _spirit.Parameters_EMA_Set_N_Mode_Follow
_EMA_Set_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
_EMA_Set_N_Mode_Follow.restype  = None
def set_n_mode_follow(p_state, n_mode, idx_image=-1, idx_chain=-1):
    _EMA_Set_N_Mode_Follow(ctypes.c_void_p(p_state), ctypes.c_int(n_mode),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

## ---------------------------------- Get ----------------------------------

### Get number of modes
_EMA_Get_N_Modes          = _spirit.Parameters_EMA_Get_N_Modes
_EMA_Get_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_EMA_Get_N_Modes.restype  = ctypes.c_int
def get_n_modes(p_state, idx_image=-1, idx_chain=-1):
    return int(_EMA_Get_N_Modes(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get index of mode to follow
_EMA_Get_N_Mode_Follow          = _spirit.Parameters_EMA_Get_N_Mode_Follow
_EMA_Get_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_EMA_Get_N_Mode_Follow.restype  = ctypes.c_int
def get_n_mode_follow(p_state, idx_image=-1, idx_chain=-1):
    return int(_EMA_Get_N_Mode_Follow(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))