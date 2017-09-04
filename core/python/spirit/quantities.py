import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### Get Chain index
_Get_Magnetization          = _spirit.Quantity_Get_Magnetization
_Get_Magnetization.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                               ctypes.c_int, ctypes.c_int]
_Get_Magnetization.restype  = None
def Get_Magnetization(p_state, idx_image=-1, idx_chain=-1):
    magnetization = (3*ctypes.c_float)()
    _Get_Magnetization(ctypes.c_void_p(p_state), magnetization,
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [float(i) for i in magnetization]
