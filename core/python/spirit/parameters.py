import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### ------------------- Set LLG -------------------

### Set LLG N Iterations
_Set_LLG_N_Iterations             = _spirit.Parameters_Set_LLG_N_Iterations
_Set_LLG_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                     ctypes.c_int, ctypes.c_int]
_Set_LLG_N_Iterations.restype     = None
def Set_LLG_N_Iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_LLG_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations), 
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), 
                          ctypes.c_int(idx_chain))

### Set LLG Direct Minimization
_Set_LLG_Direct_Minimization            = _spirit.Parameters_Set_LLG_Direct_Minimization
_Set_LLG_Direct_Minimization.argtypes   = [ctypes.c_void_p, ctypes.c_bool, 
                                           ctypes.c_int, ctypes.c_int ]
_Set_LLG_Direct_Minimization.restype    = None
def Set_LLG_Direct_Minimization(p_state, use_minimization, idx_image=-1, idx_chain=-1):
    _Set_LLG_Direct_Minimization(ctypes.c_void_p(p_state), ctypes.c_bool(use_minimization), 
                                 ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_LLG_Convergence            = _spirit.Parameters_Set_LLG_Convergence
_Set_LLG_Convergence.argtypes   = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int ]
_Set_LLG_Convergence.restype    = None
def Set_LLG_Convergence(p_state, convergence, idx_image=-1, idx_chain=-1):
    _Set_LLG_Convergence(ctypes.c_void_p(p_state), ctypes.c_float(convergence),
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set LLG Time Step
_Set_LLG_Time_Step             = _spirit.Parameters_Set_LLG_Time_Step
_Set_LLG_Time_Step.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_LLG_Time_Step.restype     = None
def Set_LLG_Time_Step(p_state, dt, idx_image=-1, idx_chain=-1):
    _Set_LLG_Time_Step(p_state, ctypes.c_float(dt), idx_image, idx_chain)

### Set LLG Damping
_Set_LLG_Damping             = _spirit.Parameters_Set_LLG_Damping
_Set_LLG_Damping.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_LLG_Damping.restype     = None
def Set_LLG_Damping(p_state, damping, idx_image=-1, idx_chain=-1):
    _Set_LLG_Damping(ctypes.c_void_p(p_state), ctypes.c_float(damping), 
                     ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set spin transfer torque
# Note: direction must be  list
_Set_LLG_STT             = _spirit.Parameters_Set_LLG_STT
_Set_LLG_STT.argtypes    = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float, 
                            ctypes.POINTER(3*ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Set_LLG_STT.restype     = None
def Set_LLG_STT(p_state, use_gradient, magnitude, direction, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    direction = vec3(*direction)
    ctypes.cast( direction, ctypes.POINTER(vec3))
    _Set_LLG_STT(ctypes.c_void_p(p_state), ctypes.c_bool(use_gradient), 
                 ctypes.c_float(magnitude), direction, ctypes.c_int(idx_image), 
                 ctypes.c_int(idx_chain))

### Set Temperature
_Set_LLG_Temperature             = _spirit.Parameters_Set_LLG_Temperature
_Set_LLG_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_LLG_Temperature.restype     = None
def Set_LLG_Temperature(p_state, temperature, idx_image=-1, idx_chain=-1):
    _Set_LLG_Temperature(ctypes.c_void_p(p_state), ctypes.c_float(temperature), 
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### ------------------- Set GNEB -------------------

### Set GNEB N Iterations
_Set_GNEB_N_Iterations             = _spirit.Parameters_Set_GNEB_N_Iterations
_Set_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                      ctypes.c_int, ctypes.c_int]
_Set_GNEB_N_Iterations.restype     = None
def Set_GNEB_N_Iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_GNEB_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations), 
                           ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), 
                           ctypes.c_int(idx_chain))

### Set GNEB convergence
_Set_GNEB_Convergence           = _spirit.Parameters_Set_GNEB_Convergence
_Set_GNEB_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Convergence.restype   = None
def Set_GNEB_Convergence(p_state, convergence, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Convergence(ctypes.c_void_p(p_state), ctypes.c_float(convergence), 
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB Spring Constant
_Set_GNEB_Spring_Constant           = _spirit.Parameters_Set_GNEB_Spring_Constant
_Set_GNEB_Spring_Constant.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Spring_Constant.restype   = None
def Set_GNEB_Spring_Constant(p_state, c_spring, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Spring_Constant(ctypes.c_void_p(p_state), ctypes.c_float(c_spring), 
                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB climbing and falling images
_Set_GNEB_Climbing_Falling             = _spirit.Parameters_Set_GNEB_Climbing_Falling
_Set_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Climbing_Falling.restype     = None
def Set_GNEB_Climbing_Falling(p_state, image_type, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Climbing_Falling(ctypes.c_void_p(p_state), ctypes.c_int(image_type),
                               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB climbing and falling images
_Set_GNEB_Image_Type_Automatically             = _spirit.Parameters_Set_GNEB_Image_Type_Automatically
_Set_GNEB_Image_Type_Automatically.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Set_GNEB_Image_Type_Automatically.restype     = None
def Set_GNEB_Image_Type_Automatically(p_state, idx_chain=-1):
    _Set_GNEB_Image_Type_Automatically(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))

### ------------------- Get LLG -------------------

### Get LLG N Iterations
_Get_LLG_N_Iterations             = _spirit.Parameters_Get_LLG_N_Iterations
_Get_LLG_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ), 
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_Get_LLG_N_Iterations.restype     = None
def Get_LLG_N_Iterations(p_state, idx_image=-1, idx_chain=-1):
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _Get_LLG_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log), 
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

### Get LLG Direct Minimization
_Get_LLG_Direct_Minimization            = _spirit.Parameters_Get_LLG_Direct_Minimization
_Get_LLG_Direct_Minimization.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int ]
_Get_LLG_Direct_Minimization.restype    = ctypes.c_bool
def Get_LLG_Direct_Minimization(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_LLG_Direct_Minimization(ctypes.c_void_p(p_state), 
                                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get LLG Convergence
_Get_LLG_Convergence            = _spirit.Parameters_Get_LLG_Convergence
_Get_LLG_Convergence.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int ]
_Get_LLG_Convergence.restype    = ctypes.c_float
def Get_LLG_Convergence(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_LLG_Convergence(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                      ctypes.c_int(idx_chain)))

### Get LLG Time Step in [psec]
_Get_LLG_Time_Step             = _spirit.Parameters_Get_LLG_Time_Step
_Get_LLG_Time_Step.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_LLG_Time_Step.restype     = ctypes.c_float
def Get_LLG_Time_Step(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_LLG_Time_Step(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                    ctypes.c_int(idx_chain)))

### Get LLG Damping
_Get_LLG_Damping             = _spirit.Parameters_Get_LLG_Damping
_Get_LLG_Damping.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_LLG_Damping.restype     = ctypes.c_float
def Get_LLG_Damping(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_LLG_Damping(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                            ctypes.c_int(idx_chain)))

### Get LLG STT
_Get_LLG_STT             = _spirit.Parameters_Get_LLG_STT
_Get_LLG_STT.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_bool ), 
                            ctypes.POINTER( ctypes.c_float ), ctypes.POINTER(ctypes.c_float), 
                            ctypes.c_int, ctypes.c_int]
_Get_LLG_STT.restype     = None
def Get_LLG_STT(p_state, idx_image=-1, idx_chain=-1):
    direction = (3*ctypes.c_float)()
    use_gradient = ctypes.c_bool()
    magnitude = ctypes.c_float()
    _Get_LLG_STT(ctypes.c_void_p(p_state), ctypes.pointer(use_gradient), 
                 ctypes.pointer(magnitude), direction, ctypes.c_int(idx_image), 
                 ctypes.c_int(idx_chain))
    return float(magnitude.value), [float(i) for i in direction], bool(use_gradient)
    
### Get Temperature
_Get_LLG_Temperature             = _spirit.Parameters_Get_LLG_Temperature
_Get_LLG_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_LLG_Temperature.restype     = ctypes.c_float
def Get_LLG_Temperature(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_LLG_Temperature(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                     ctypes.c_int(idx_chain)))

### ------------------- Get GNEB -------------------

### Get GNEB N Iterations
_Get_GNEB_N_Iterations             = _spirit.Parameters_Get_GNEB_N_Iterations
_Get_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ), 
                                      ctypes.POINTER( ctypes.c_int ), ctypes.c_int]
_Get_GNEB_N_Iterations.restype     = None
def Get_GNEB_N_Iterations(p_state, idx_image=-1, idx_chain=-1):
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _Get_GNEB_N_Iterations(ctypes.c_void_p(p_state), ctypes.pointer(n_iterations), 
                           ctypes.pointer(n_iterations_log), ctypes.c_int(idx_image), 
                           ctypes.c_int(idx_chain) )
    return int(n_iterations.value), int(n_iterations_log.value)

### Get GNEB convergence
_Get_GNEB_Convergence           = _spirit.Parameters_Get_GNEB_Convergence
_Get_GNEB_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_GNEB_Convergence.restype   = ctypes.c_float
def Get_GNEB_Convergence(p_state, idx_image=-1, idx_chain=-1):
    return float( _Get_GNEB_Convergence(ctypes.c_void_p(p_state), 
                                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB Spring Constant
_Get_GNEB_Spring_Constant             = _spirit.Parameters_Get_GNEB_Spring_Constant
_Get_GNEB_Spring_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_GNEB_Spring_Constant.restype     = ctypes.c_int
def Get_GNEB_Spring_Constant(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_GNEB_Spring_Constant(ctypes.c_void_p(p_state), 
                                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB climbing and falling images
_Get_GNEB_Climbing_Falling             = _spirit.Parameters_Get_GNEB_Climbing_Falling
_Get_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_GNEB_Climbing_Falling.restype     = ctypes.c_int
def Get_GNEB_Climbing_Falling(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_GNEB_Climbing_Falling(ctypes.c_void_p(p_state), 
                                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB number of energy interpolations
_Get_GNEB_N_Energy_Interpolations             = _spirit.Parameters_Get_GNEB_N_Energy_Interpolations
_Get_GNEB_N_Energy_Interpolations.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Get_GNEB_N_Energy_Interpolations.restype     = ctypes.c_int
def Get_GNEB_N_Energy_Interpolations(p_state, idx_chain=-1):
    return int(_Get_GNEB_N_Energy_Interpolations(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))