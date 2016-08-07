import corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Domain Wall configuration
_Homogeneous             = _core.Transition_Homogeneous
_Homogeneous.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Homogeneous.restype     = None
def Homogeneous(p_state, idx_1, idx_2, idx_chain=-1):
    vec3 = ctypes.c_double * 3
    _Homogeneous(p_state, idx_1, idx_2, idx_chain)


