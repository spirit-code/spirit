import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Domain (homogeneous) configuration
_Domain             = _spirit.Configuration_Domain
_Domain.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                       ctypes.c_float, ctypes.c_float, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Domain.restype     = None
def Domain(p_state, dir, pos=[0,0,0], border_rectangular=[-1,-1,-1], border_cylindrical=-1, 
           border_spherical=-1, inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Domain(ctypes.c_void_p(p_state) , vec3(*dir), vec3(*pos), vec3(*border_rectangular), 
            ctypes.c_float(border_cylindrical), ctypes.c_float(border_spherical), 
            ctypes.c_bool(inverted), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### All spins in +z direction
_PlusZ             = _spirit.Configuration_PlusZ
_PlusZ.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                      ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_bool,
                      ctypes.c_int, ctypes.c_int]
_PlusZ.restype     = None
def PlusZ(p_state, pos=[0.0,0.0,0.0], border_rectangular=[-1.0,-1.0,-1.0], border_cylindrical=-1.0, 
          border_spherical=-1.0, inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _PlusZ(ctypes.c_void_p(p_state) , vec3(*pos), vec3(*border_rectangular), 
           ctypes.c_float(border_cylindrical), ctypes.c_float(border_spherical), 
           ctypes.c_bool(inverted), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### All spins in -z direction
_MinusZ             = _spirit.Configuration_MinusZ
_MinusZ.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                       ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, 
                       ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MinusZ.restype     = None
def MinusZ(p_state, pos=[0,0,0], border_rectangular=[-1,-1,-1], border_cylindrical=-1, 
          border_spherical=-1, inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _MinusZ(ctypes.c_void_p(p_state) , vec3(*pos), vec3(*border_rectangular), 
            ctypes.c_float(border_cylindrical), ctypes.c_float(border_spherical), 
            ctypes.c_bool(inverted), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Random configuration
_Random             = _spirit.Configuration_Random
_Random.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
                       ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, 
                       ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Random.restype     = None
def Random(p_state, pos=[0,0,0], border_rectangular=[-1,-1,-1], border_cylindrical=-1, 
           border_spherical=-1, inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Random(ctypes.c_void_p(p_state) , vec3(*pos), vec3(*border_rectangular), 
            ctypes.c_float(border_cylindrical), ctypes.c_float(border_spherical), 
            ctypes.c_bool(inverted), False, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### Add temperature-scaled random noise to configuration
_Add_Noise_Temperature             = _spirit.Configuration_Add_Noise_Temperature
_Add_Noise_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, 
                                      ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                                      ctypes.c_float, ctypes.c_float, ctypes.c_bool, ctypes.c_bool, 
                                      ctypes.c_int, ctypes.c_int]
_Add_Noise_Temperature.restype     = None
def Add_Noise_Temperature(p_state, temperature, pos=[0,0,0], border_rectangular=[-1,-1,-1], 
                          border_cylindrical=-1, border_spherical=-1, inverted=False, 
                          idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Add_Noise_Temperature(ctypes.c_void_p(p_state) , ctypes.c_float(temperature), vec3(*pos), 
                           vec3(*border_rectangular), ctypes.c_float(border_cylindrical), 
                           ctypes.c_float(border_spherical), ctypes.c_bool(inverted), False, 
                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### Skyrmion configuration
_Skyrmion             = _spirit.Configuration_Skyrmion
_Skyrmion.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, 
                         ctypes.c_bool, ctypes.c_bool, ctypes.c_bool, ctypes.POINTER(ctypes.c_float), 
                         ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_bool, 
                         ctypes.c_int, ctypes.c_int]
_Skyrmion.restype     = None
def Skyrmion(p_state, radius, order=1, phase=1, upDown=False, achiral=False, rightleft=False, 
             pos=[0,0,0], border_rectangular=[-1,-1,-1], border_cylindrical=-1, border_spherical=-1, 
             inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Skyrmion(ctypes.c_void_p(p_state) , ctypes.c_float(radius), ctypes.c_float(order), 
              ctypes.c_float(phase), ctypes.c_bool(upDown), ctypes.c_bool(achiral), 
              ctypes.c_bool(rightleft), vec3(*pos), vec3(*border_rectangular), 
              ctypes.c_float(border_cylindrical), ctypes.c_float(border_spherical), 
              ctypes.c_bool(inverted), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Hopfion configuration
_Hopfion             = _spirit.Configuration_Hopfion
_Hopfion.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float), 
                        ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_bool, 
                        ctypes.c_int, ctypes.c_int]
_Hopfion.restype     = None
def Hopfion(p_state, radius, order=1, pos=[0,0,0], border_rectangular=[-1,-1,-1], 
            border_cylindrical=-1, border_spherical=-1, inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Hopfion(ctypes.c_void_p(p_state) , ctypes.c_float(radius), ctypes.c_int(order), 
             vec3(*pos), vec3(*border_rectangular), ctypes.c_float(border_cylindrical), 
             ctypes.c_float(border_spherical), ctypes.c_bool(inverted), 
             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### Spin Spiral configuration
_SpinSpiral             = _spirit.Configuration_SpinSpiral
_SpinSpiral.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), 
                           ctypes.POINTER(ctypes.c_float), ctypes.c_float, 
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), 
                           ctypes.c_float, ctypes.c_float, ctypes.c_bool, 
                           ctypes.c_int, ctypes.c_int]
_SpinSpiral.restype     = None
def SpinSpiral(p_state, direction_type, q_vector, axis, theta, pos=[0,0,0], 
               border_rectangular=[-1,-1,-1], border_cylindrical=-1, border_spherical=-1, 
               inverted=False, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _SpinSpiral(ctypes.c_void_p(p_state) , ctypes.c_char_p(direction_type.encode('utf-8')), 
                vec3(*q_vector), vec3(*axis), ctypes.c_float(theta), vec3(*pos), 
                vec3(*border_rectangular), ctypes.c_float(border_cylindrical), 
                ctypes.c_float(border_spherical), ctypes.c_bool(inverted), 
                ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
