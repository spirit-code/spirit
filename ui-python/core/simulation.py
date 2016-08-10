import corelib
import ctypes
import threading

### Load Library
_core = corelib.LoadCoreLibrary()


### Setup State
_PlayPause          = _core.Simulation_PlayPause
_PlayPause.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_PlayPause.restype  = None
def PlayPause(p_state, method_type, optimizer_type, n_iterations=-1, log_steps=-1, idx_image=-1, idx_chain=-1):
    _PlayPause(p_state, ctypes.c_char_p(method_type), ctypes.c_char_p(optimizer_type), ctypes.c_int(n_iterations), ctypes.c_int(log_steps), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Setup State
def ThreadedPlayPause(p_state, method_type, optimizer_type, n_iterations=-1, log_steps=-1, idx_image=-1, idx_chain=-1):
    t = threading.Thread(target=PlayPause, args=[p_state, b"LLG", b"SIB", n_iterations, log_steps, idx_image, idx_chain])
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exit
        t.join(.1)

### Stop All
_Stop_All           = _core.Simulation_Stop_All
_Stop_All.argtypes  = [ctypes.c_void_p]
_Stop_All.restype   = None
def Stop_All(p_state):
    _Stop_All(p_state)

    
### Check if LLG running
_Running_LLG            = _core.Simulation_Running_LLG
_Running_LLG.argtypes   = [ctypes.c_void_p]
_Running_LLG.restype    = ctypes.c_bool
def Running_LLG(p_state):
    return bool(_Running_LLG(p_state))


### Check if GNEB running
_Running_GNEB           = _core.Simulation_Running_GNEB
_Running_GNEB.argtypes  = [ctypes.c_void_p]
_Running_GNEB.restype   = ctypes.c_bool
def Running_GNEB(p_state):
    return bool(_Running_GNEB(p_state))


### Check if MMF running
_Running_MMF            = _core.Simulation_Running_MMF
_Running_MMF.argtypes   = [ctypes.c_void_p]
_Running_MMF.restype    = ctypes.c_bool
def Running_MMF(p_state):
    return bool(_Running_MMF(p_state))