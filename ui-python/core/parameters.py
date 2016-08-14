import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### TODO:
# all getters

### Set LLG Time Step
_Set_LLG_Time_Step             = _core.Parameters_Set_LLG_Time_Step
_Set_LLG_Time_Step.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_LLG_Time_Step.restype     = None
def Set_LLG_Time_Step(p_state, dt, idx_image, idx_chain):
    _Set_LLG_Time_Step(p_state, ctypes.c_float(dt), idx_image, idx_chain)

### Set LLG Damping
_Set_LLG_Damping             = _core.Parameters_Set_LLG_Damping
_Set_LLG_Damping.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_LLG_Damping.restype     = None
def Set_LLG_Damping(p_state, damping, idx_image, idx_chain):
    _Set_LLG_Damping(p_state, ctypes.c_float(damping), idx_image, idx_chain)

### Set LLG N Iterations
_Set_LLG_N_Iterations             = _core.Parameters_Set_LLG_N_Iterations
_Set_LLG_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_LLG_N_Iterations.restype     = None
def Set_LLG_N_Iterations(p_state, n_iterations, idx_image, idx_chain):
    _Set_LLG_N_Iterations(p_state, ctypes.c_int(n_iterations), idx_image, idx_chain)

### Set LLG Log Steps
_Set_LLG_Log_Steps             = _core.Parameters_Set_LLG_Log_Steps
_Set_LLG_Log_Steps.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_LLG_Log_Steps.restype     = None
def Set_LLG_Log_Steps(p_state, log_steps, idx_image, idx_chain):
    _Set_LLG_Log_Steps(p_state, ctypes.c_int(log_steps), idx_image, idx_chain)


### Set GNEB Spring Constant
_Set_GNEB_Spring_Constant             = _core.Parameters_Set_GNEB_Spring_Constant
_Set_GNEB_Spring_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Spring_Constant.restype     = None
def Set_GNEB_Spring_Constant(p_state, c_spring, idx_image, idx_chain):
    _Set_GNEB_Spring_Constant(p_state, ctypes.c_float(c_spring), idx_image, idx_chain)

### Set GNEB climbing and falling images
_Set_GNEB_Climbing_Falling             = _core.Parameters_Set_GNEB_Climbing_Falling
_Set_GNEB_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Climbing_Falling.restype     = None
def Set_GNEB_Climbing_Falling(p_state, damping, idx_image, idx_chain):
    _Set_GNEB_Climbing_Falling(p_state, ctypes.c_float(damping), idx_image, idx_chain)

### Set GNEB N Iterations
_Set_GNEB_N_Iterations             = _core.Parameters_Set_GNEB_N_Iterations
_Set_GNEB_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_N_Iterations.restype     = None
def Set_GNEB_N_Iterations(p_state, n_iterations, idx_image, idx_chain):
    _Set_GNEB_N_Iterations(p_state, ctypes.c_int(n_iterations), idx_image, idx_chain)

### Set GNEB Log Steps
_Set_GNEB_Log_Steps             = _core.Parameters_Set_GNEB_Log_Steps
_Set_GNEB_Log_Steps.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_GNEB_Log_Steps.restype     = None
def Set_GNEB_Log_Steps(p_state, log_steps, idx_image, idx_chain):
    _Set_GNEB_Log_Steps(p_state, ctypes.c_int(log_steps), idx_image, idx_chain)