import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### ------------------- Set GNEB -------------------

### Set GNEB N Iterations
_Set_GNEB_N_Iterations             = _spirit.Parameters_Set_GNEB_N_Iterations
_Set_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                      ctypes.c_int, ctypes.c_int]
_Set_GNEB_N_Iterations.restype     = None
def setIterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_GNEB_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations), 
                           ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), 
                           ctypes.c_int(idx_chain))

### Set GNEB convergence
_Set_GNEB_Convergence           = _spirit.Parameters_Set_GNEB_Convergence
_Set_GNEB_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Convergence.restype   = None
def setConvergence(p_state, convergence, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Convergence(ctypes.c_void_p(p_state), ctypes.c_float(convergence), 
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB Spring Constant
_Set_GNEB_Spring_Constant           = _spirit.Parameters_Set_GNEB_Spring_Constant
_Set_GNEB_Spring_Constant.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Spring_Constant.restype   = None
def setSpringConstant(p_state, c_spring, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Spring_Constant(ctypes.c_void_p(p_state), ctypes.c_float(c_spring), 
                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB climbing and falling images
_Set_GNEB_Climbing_Falling             = _spirit.Parameters_Set_GNEB_Climbing_Falling
_Set_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Climbing_Falling.restype     = None
def setClimbingFalling(p_state, image_type, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Climbing_Falling(ctypes.c_void_p(p_state), ctypes.c_int(image_type),
                               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB climbing and falling images
_Set_GNEB_Image_Type_Automatically             = _spirit.Parameters_Set_GNEB_Image_Type_Automatically
_Set_GNEB_Image_Type_Automatically.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Set_GNEB_Image_Type_Automatically.restype     = None
def setImageTypeAutomatically(p_state, idx_chain=-1):
    _Set_GNEB_Image_Type_Automatically(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))

### ------------------- Get GNEB -------------------

### Get GNEB N Iterations
_Get_GNEB_N_Iterations             = _spirit.Parameters_Get_GNEB_N_Iterations
_Get_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ), 
                                      ctypes.POINTER( ctypes.c_int ), ctypes.c_int]
_Get_GNEB_N_Iterations.restype     = None
def getIterations(p_state, idx_image=-1, idx_chain=-1):
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
def getConvergence(p_state, idx_image=-1, idx_chain=-1):
    return float( _Get_GNEB_Convergence(ctypes.c_void_p(p_state), 
                                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB Spring Constant
_Get_GNEB_Spring_Constant             = _spirit.Parameters_Get_GNEB_Spring_Constant
_Get_GNEB_Spring_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_GNEB_Spring_Constant.restype     = ctypes.c_float
def getSpringConstant(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_GNEB_Spring_Constant(ctypes.c_void_p(p_state), 
                                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB climbing and falling images
_Get_GNEB_Climbing_Falling             = _spirit.Parameters_Get_GNEB_Climbing_Falling
_Get_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_GNEB_Climbing_Falling.restype     = ctypes.c_int
def getClimbingFalling(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_GNEB_Climbing_Falling(ctypes.c_void_p(p_state), 
                                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB number of energy interpolations
_Get_GNEB_N_Energy_Interpolations             = _spirit.Parameters_Get_GNEB_N_Energy_Interpolations
_Get_GNEB_N_Energy_Interpolations.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Get_GNEB_N_Energy_Interpolations.restype     = ctypes.c_int
def getEnergyInterpolations(p_state, idx_chain=-1):
    return int(_Get_GNEB_N_Energy_Interpolations(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))