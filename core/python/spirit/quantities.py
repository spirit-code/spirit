import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Get Chain index
_Get_Magnetization          = _spirit.Quantity_Get_Magnetization
_Get_Magnetization.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Magnetization.restype  = None
def Get_Magnetization(p_state, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    M = [0,0,0]
    _M = vec3(*M)
    _Get_Magnetization(p_state, _M, idx_image, idx_chain)
    for i in range(3):
        M[i] = _M[i]
    return M