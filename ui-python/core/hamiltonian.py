import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Domain Wall configuration
_Set_STT             = _core.Hamiltonian_Set_STT
_Set_STT.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
_Set_STT.restype     = None
def Set_STT(p_state, magnitude, direction):
    vec3 = ctypes.c_double * 3
    _Set_STT(p_state, ctypes.c_float(magnitude), direction[0], direction[1], direction[2])

