import corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()

### Setup State
_PlayPause = _core.Simulation_PlayPause
_PlayPause.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_PlayPause.restype = None
def PlayPause(p_state, method_type, optimizer_type, idx_image, idx_chain):
    global _core
    return _core.Simulation_PlayPause(p_state, ctypes.c_char_p(method_type), ctypes.c_char_p(optimizer_type), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))