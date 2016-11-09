import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Domain Wall configuration
_DomainWall             = _core.Configuration_DomainWall
_DomainWall.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_DomainWall.restype     = None
def DomainWall(p_state, pos, dir, greater, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _DomainWall(p_state, vec3(*pos), vec3(*dir), ctypes.c_bool(greater), idx_image, idx_chain)


### Homogeneous configuration
_Homogeneous             = _core.Configuration_Homogeneous
_Homogeneous.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Homogeneous.restype     = None
def Homogeneous(p_state, dir, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Homogeneous(p_state, vec3(*dir), idx_image, idx_chain)


### All spins in +z direction
_PlusZ             = _core.Configuration_PlusZ
_PlusZ.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_PlusZ.restype     = None
def PlusZ(p_state, idx_image=-1, idx_chain=-1):
    _PlusZ(p_state, idx_image, idx_chain)


### All spins in -z direction
_MinusZ             = _core.Configuration_MinusZ
_MinusZ.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_MinusZ.restype     = None
def MinusZ(p_state, idx_image=-1, idx_chain=-1):
    _MinusZ(p_state, idx_image, idx_chain)


### Random configuration
_Random             = _core.Configuration_Random
_Random.argtypes    = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Random.restype     = None
def Random(p_state, external=False, idx_image=-1, idx_chain=-1):
    _Random(p_state, idx_image, idx_chain)


### Add temperature-scaled random noise to configuration
_Add_Noise_Temperature             = _core.Configuration_Add_Noise_Temperature
_Add_Noise_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Add_Noise_Temperature.restype     = None
def Add_Noise_Temperature(p_state, temperature, idx_image=-1, idx_chain=-1):
    _Add_Noise_Temperature(p_state, temperature, idx_image, idx_chain)


### Skyrmion configuration
_Skyrmion             = _core.Configuration_Skyrmion
_Skyrmion.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Skyrmion.restype     = None
def Skyrmion(p_state, pos, radius, order, phase, upDown, achiral, rightleft, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Skyrmion(p_state, vec3(*pos), radius, order, phase, ctypes.c_bool(upDown), ctypes.c_bool(achiral), ctypes.c_bool(rightleft), idx_image, idx_chain)


### Spin Spiral configuration
_SpinSpiral             = _core.Configuration_SpinSpiral
_SpinSpiral.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_int, ctypes.c_int]
_SpinSpiral.restype     = None
def SpinSpiral(p_state, direction_type, q_vector, axis, theta, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _SpinSpiral(p_state, ctypes.c_char_p(direction_type), vec3(*q_vector), vec3(*axis), theta, idx_image, idx_chain)