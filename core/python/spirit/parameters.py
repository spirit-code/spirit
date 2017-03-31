import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### ------------------- Set LLG -------------------
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
    _Set_LLG_Damping(p_state, ctypes.c_float(damping), idx_image, idx_chain)

### Set LLG N Iterations
_Set_LLG_N_Iterations             = _spirit.Parameters_Set_LLG_N_Iterations
_Set_LLG_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_LLG_N_Iterations.restype     = None
def Set_LLG_N_Iterations(p_state, n_iterations, idx_image=-1, idx_chain=-1):
    _Set_LLG_N_Iterations(p_state, ctypes.c_int(n_iterations), idx_image, idx_chain)

### Set LLG Log Steps
_Set_LLG_N_Iterations_Log             = _spirit.Parameters_Set_LLG_N_Iterations_Log
_Set_LLG_N_Iterations_Log.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_LLG_N_Iterations_Log.restype     = None
def Set_LLG_N_Iterations_Log(p_state, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_LLG_N_Iterations_Log(p_state, ctypes.c_int(n_iterations_log), idx_image, idx_chain)

### ------------------- Set GNEB -------------------
### Set GNEB Spring Constant
_Set_GNEB_Spring_Constant             = _spirit.Parameters_Set_GNEB_Spring_Constant
_Set_GNEB_Spring_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Spring_Constant.restype     = None
def Set_GNEB_Spring_Constant(p_state, c_spring, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Spring_Constant(p_state, ctypes.c_float(c_spring), idx_image, idx_chain)

### Set GNEB climbing and falling images
_Set_GNEB_Climbing_Falling             = _spirit.Parameters_Set_GNEB_Climbing_Falling
_Set_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Climbing_Falling.restype     = None
def Set_GNEB_Climbing_Falling(p_state, image_type, idx_image=-1, idx_chain=-1):
    _Set_GNEB_Climbing_Falling(p_state, image_type, idx_image, idx_chain)

### Set GNEB N Iterations
_Set_GNEB_N_Iterations             = _spirit.Parameters_Set_GNEB_N_Iterations
_Set_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_N_Iterations.restype     = None
def Set_GNEB_N_Iterations(p_state, n_iterations, idx_image=-1, idx_chain=-1):
    _Set_GNEB_N_Iterations(p_state, ctypes.c_int(n_iterations), idx_image, idx_chain)

### Set GNEB Log Steps
_Set_GNEB_N_Iterations_Log             = _spirit.Parameters_Set_GNEB_N_Iterations_Log
_Set_GNEB_N_Iterations_Log.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_N_Iterations_Log.restype     = None
def Set_GNEB_N_Iterations_Log(p_state, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_GNEB_N_Iterations_Log(p_state, ctypes.c_int(n_iterations_log), idx_image, idx_chain)


### ------------------- Get LLG -------------------
### Get LLG Time Step
_Get_LLG_Time_Step             = _spirit.Parameters_Get_LLG_Time_Step
_Get_LLG_Time_Step.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_LLG_Time_Step.restype     = None
def Get_LLG_Time_Step(p_state, dt, idx_image=-1, idx_chain=-1):
    _Get_LLG_Time_Step(p_state, ctypes.c_float(dt), idx_image, idx_chain)

### Get LLG Damping
_Get_LLG_Damping             = _spirit.Parameters_Get_LLG_Damping
_Get_LLG_Damping.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_LLG_Damping.restype     = None
def Get_LLG_Damping(p_state, damping, idx_image=-1, idx_chain=-1):
    _Get_LLG_Damping(p_state, ctypes.c_float(damping), idx_image, idx_chain)

### Get LLG N Iterations
_Get_LLG_N_Iterations             = _spirit.Parameters_Get_LLG_N_Iterations
_Get_LLG_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_LLG_N_Iterations.restype     = ctypes.c_int
def Get_LLG_N_Iterations(p_state, n_iterations, idx_image=-1, idx_chain=-1):
    return _Get_LLG_N_Iterations(p_state, ctypes.c_int(n_iterations), idx_image, idx_chain)

### Get LLG Log Steps
_Get_LLG_N_Iterations_Log             = _spirit.Parameters_Get_LLG_N_Iterations_Log
_Get_LLG_N_Iterations_Log.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_LLG_N_Iterations_Log.restype     = ctypes.c_int
def Get_LLG_N_Iterations_Log(p_state, n_iterations_log, idx_image=-1, idx_chain=-1):
    return _Get_LLG_N_Iterations_Log(p_state, ctypes.c_int(n_iterations_log), idx_image, idx_chain)

### ------------------- Get GNEB -------------------
### Get GNEB Spring Constant
_Get_GNEB_Spring_Constant             = _spirit.Parameters_Get_GNEB_Spring_Constant
_Get_GNEB_Spring_Constant.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_GNEB_Spring_Constant.restype     = None
def Get_GNEB_Spring_Constant(p_state, c_spring, idx_image=-1, idx_chain=-1):
    _Get_GNEB_Spring_Constant(p_state, ctypes.c_float(c_spring), idx_image, idx_chain)

### Get GNEB climbing and falling images
_Get_GNEB_Climbing_Falling             = _spirit.Parameters_Get_GNEB_Climbing_Falling
_Get_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int]
_Get_GNEB_Climbing_Falling.restype     = None
def Get_GNEB_Climbing_Falling(p_state, climbing, falling, idx_image=-1, idx_chain=-1):
    _Get_GNEB_Climbing_Falling(p_state, ctypes.POINTER(ctypes.c_bool(climbing)), ctypes.POINTER(ctypes.c_bool(falling)), idx_image, idx_chain)

### Get GNEB N Iterations
_Get_GNEB_N_Iterations             = _spirit.Parameters_Get_GNEB_N_Iterations
_Get_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_GNEB_N_Iterations.restype     = ctypes.c_int
def Get_GNEB_N_Iterations(p_state, idx_chain=-1):
    return _Get_GNEB_N_Iterations(p_state, idx_chain)

### Get GNEB Log Steps
_Get_GNEB_N_Iterations_Log             = _spirit.Parameters_Get_GNEB_N_Iterations_Log
_Get_GNEB_N_Iterations_Log.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Get_GNEB_N_Iterations_Log.restype     = ctypes.c_int
def Get_GNEB_N_Iterations_Log(p_state, idx_chain=-1):
    return _Get_GNEB_N_Iterations_Log(p_state, idx_chain)

    
### Get GNEB number of energy interpolations
_Get_GNEB_N_Energy_Interpolations             = _spirit.Parameters_Get_GNEB_N_Energy_Interpolations
_Get_GNEB_N_Energy_Interpolations.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Get_GNEB_N_Energy_Interpolations.restype     = ctypes.c_int
def Get_GNEB_N_Energy_Interpolations(p_state, idx_chain=-1):
    return _Get_GNEB_N_Energy_Interpolations(p_state, idx_chain)