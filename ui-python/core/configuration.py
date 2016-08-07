import corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Copy active image to clipboard
_DomainWall             = _core.Configuration_DomainWall
_DomainWall.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_DomainWall.restype     = None
def DomainWall(p_state, pos, dir, greater, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_double * 3
    _DomainWall(p_state, vec3(*pos), vec3(*dir), ctypes.c_bool(greater), idx_image, idx_chain)


### Copy active image to clipboard
_Homogeneous             = _core.Configuration_Homogeneous
_Homogeneous.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
_Homogeneous.restype     = None
def Homogeneous(p_state, dir, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_double * 3
    _Homogeneous(p_state, vec3(*dir), idx_image, idx_chain)


# Configuration_PlusZ

# Configuration_MinusZ

# Configuration_Random

# Configuration_Skyrmion

# Configuration_SpinSpiral