import os 
import ctypes

### Get this file's directory. The library should be here
core_dir = os.path.dirname(os.path.realpath(__file__))
### Load the core library
_core = ctypes.CDLL(core_dir + '/libcore.so')

### Setup State
_PlayPause = _core.Simulation_PlayPause
_PlayPause.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_PlayPause.restype = None
def PlayPause(p_state, method_type, optimizer_type, idx_image, idx_chain):
    global _core
    return _core.Simulation_PlayPause(p_state, ctypes.c_char_p(method_type), ctypes.c_char_p(optimizer_type), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))