import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

import threading

###     We use a thread for PlayPause, so that KeyboardInterrupt can be forwarded to the CDLL call
###     We might want to think about using PyDLL and about a signal handler in the core library
###     see here: http://stackoverflow.com/questions/14271697/ctrlc-doesnt-interrupt-call-to-shared-library-using-ctypes-in-python

### SingleShot Iteration
_SingleShot          = _spirit.Simulation_SingleShot
_SingleShot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_SingleShot.restype  = None
def SingleShot(p_state, method_type, optimizer_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1):
    # _SingleShot(p_state, ctypes.c_char_p(method_type), ctypes.c_char_p(optimizer_type), ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    spiritlib.WrapFunction(_SingleShot, [p_state, ctypes.c_char_p(method_type.encode('utf-8')), ctypes.c_char_p(optimizer_type.encode('utf-8')), ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])

### Play/Pause
_PlayPause          = _spirit.Simulation_PlayPause
_PlayPause.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_PlayPause.restype  = None
def PlayPause(p_state, method_type, optimizer_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_PlayPause, [p_state, ctypes.c_char_p(method_type.encode('utf-8')), ctypes.c_char_p(optimizer_type.encode('utf-8')), ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    #_PlayPause(p_state, ctypes.c_char_p(method_type), ctypes.c_char_p(optimizer_type), ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Stop All
_Stop_All           = _spirit.Simulation_Stop_All
_Stop_All.argtypes  = [ctypes.c_void_p]
_Stop_All.restype   = None
def Stop_All(p_state):
    _Stop_All(p_state)


### Check if any Simulation is running on any image/chain/collection
_Running_Any_Anywhere            = _spirit.Simulation_Running_Any_Anywhere
_Running_Any_Anywhere.argtypes   = [ctypes.c_void_p]
_Running_Any_Anywhere.restype    = ctypes.c_bool
def Running_Any_Anywhere(p_state):
    return bool(_Running_Any_Anywhere(p_state))

### Check if LLG is running on any image in any chain
_Running_LLG_Anywhere            = _spirit.Simulation_Running_LLG_Anywhere
_Running_LLG_Anywhere.argtypes   = [ctypes.c_void_p]
_Running_LLG_Anywhere.restype    = ctypes.c_bool
def Running_LLG_Anywhere(p_state):
    return bool(_Running_LLG_Anywhere(p_state))

### Check if LLG running
_Running_GNEB_Anywhere            = _spirit.Simulation_Running_GNEB_Anywhere
_Running_GNEB_Anywhere.argtypes   = [ctypes.c_void_p]
_Running_GNEB_Anywhere.restype    = ctypes.c_bool
def Running_GNEB_Anywhere(p_state):
    return bool(_Running_GNEB_Anywhere(p_state))


### Check if LLG running on a chain
_Running_LLG_Chain            = _spirit.Simulation_Running_LLG_Chain
_Running_LLG_Chain.argtypes   = [ctypes.c_void_p, ctypes.c_int]
_Running_LLG_Chain.restype    = ctypes.c_bool
def Running_LLG_Chain(p_state, idx_chain=-1):
    return bool(_Running_LLG_Chain(p_state, idx_chain))


### Check if any simulation running on current image, chain or collection
_Running_Any            = _spirit.Simulation_Running_Any
_Running_Any.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Running_Any.restype    = ctypes.c_bool
def Running_Any(p_state, idx_image=-1, idx_chain=-1):
    return bool(_Running_Any(p_state, idx_image, idx_chain))


### Check if LLG running on image
_Running_LLG            = _spirit.Simulation_Running_LLG
_Running_LLG.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Running_LLG.restype    = ctypes.c_bool
def Running_LLG(p_state, idx_image=-1, idx_chain=-1):
    return bool(_Running_LLG(p_state, idx_image, idx_chain))


### Check if GNEB running on chain
_Running_GNEB           = _spirit.Simulation_Running_GNEB
_Running_GNEB.argtypes  = [ctypes.c_void_p, ctypes.c_int]
_Running_GNEB.restype   = ctypes.c_bool
def Running_GNEB(p_state, idx_chain=-1):
    return bool(_Running_GNEB(p_state, idx_chain))


### Check if MMF running
_Running_MMF            = _spirit.Simulation_Running_MMF
_Running_MMF.argtypes   = [ctypes.c_void_p]
_Running_MMF.restype    = ctypes.c_bool
def Running_MMF(p_state):
    return bool(_Running_MMF(p_state))