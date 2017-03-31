import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Get Number of Chains in Collection
_Get_NOC          = _spirit.Collection_Get_NOC
_Get_NOC.argtypes = [ctypes.c_void_p]
_Get_NOC.restype  = ctypes.c_int
def Get_NOC(p_state):
    return int(_Get_NOC(p_state))

### Go to next chain in collection
_next_Chain          = _spirit.Collection_next_Chain
_next_Chain.argtypes = [ctypes.c_void_p]
_next_Chain.restype  = ctypes.c_int
def Next_Chain(p_state):
    _next_Chain(p_state)

### Go to previous chain in collection
_prev_Chain          = _spirit.Collection_prev_Chain
_prev_Chain.argtypes = [ctypes.c_void_p]
_prev_Chain.restype  = ctypes.c_int
def Prev_Chain(p_state):
    _prev_Chain(p_state)