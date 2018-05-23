import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

_Get_Magnetization          = _spirit.Quantity_Get_Magnetization
_Get_Magnetization.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Get_Magnetization.restype  = None
def Get_Magnetization(p_state, idx_image=-1, idx_chain=-1):
    magnetization = (3*ctypes.c_float)()
    _Get_Magnetization(ctypes.c_void_p(p_state), magnetization,
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [float(i) for i in magnetization]

_Get_Topological_Charge          = _spirit.Quantity_Get_Topological_Charge
_Get_Topological_Charge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Topological_Charge.restype  = ctypes.c_float
def Get_Topological_Charge(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_Topological_Charge(ctypes.c_void_p(p_state),
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))