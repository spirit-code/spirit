import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### ------------------- Set MC -------------------

### Set number of iterations and step size
_Set_MC_N_Iterations             = _spirit.Parameters_Set_MC_N_Iterations
_Set_MC_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, 
                                     ctypes.c_int, ctypes.c_int]
_Set_MC_N_Iterations.restype     = None
def setIterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_MC_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations), 
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), 
                          ctypes.c_int(idx_chain))

### Set temperature
_Set_MC_Temperature             = _spirit.Parameters_Set_MC_Temperature
_Set_MC_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_MC_Temperature.restype     = None
def setTemperature(p_state, temperature, idx_image=-1, idx_chain=-1):
    _Set_MC_Temperature(ctypes.c_void_p(p_state), ctypes.c_float(temperature), 
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set target acceptance ratio
_Set_MC_Acceptance_Ratio             = _spirit.Parameters_Set_MC_Acceptance_Ratio
_Set_MC_Acceptance_Ratio.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_MC_Acceptance_Ratio.restype     = None
def setAcceptanceRatio(p_state, ratio, idx_image=-1, idx_chain=-1):
    _Set_MC_Acceptance_Ratio(p_state, ctypes.c_float(ratio), idx_image, idx_chain)

## ------------------- Get MC -------------------

### Get number of iterations and step size
_Get_MC_N_Iterations             = _spirit.Parameters_Get_MC_N_Iterations
_Get_MC_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ), 
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_Get_MC_N_Iterations.restype     = None
def getIterations(p_state, idx_image=-1, idx_chain=-1):
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _Get_MC_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log), 
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

### Get temperature
_Get_MC_Temperature             = _spirit.Parameters_Get_MC_Temperature
_Get_MC_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_MC_Temperature.restype     = ctypes.c_float
def getTemperature(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_MC_Temperature(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                             ctypes.c_int(idx_chain)))

### Get target acceptance ratio
_Get_MC_Acceptance_Ratio             = _spirit.Parameters_Get_MC_Acceptance_Ratio
_Get_MC_Acceptance_Ratio.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_MC_Acceptance_Ratio.restype     = ctypes.c_float
def getAcceptanceRatio(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_MC_Acceptance_Ratio(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                                    ctypes.c_int(idx_chain)))