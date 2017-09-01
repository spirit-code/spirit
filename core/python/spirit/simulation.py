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
_SingleShot.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
_SingleShot.restype  = None
def SingleShot(p_state, method_type, solver_type, n_iterations=-1, n_iterations_log=-1, 
               idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_SingleShot, [ctypes.c_void_p(p_state), 
                                         ctypes.c_char_p(method_type.encode('utf-8')), 
                                         ctypes.c_char_p(solver_type.encode('utf-8')), 
                                         ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log), 
                                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    # _SingleShot(ctypes.c_void_p(p_state), ctypes.c_char_p(method_type), 
    #             ctypes.c_char_p(solver_type), ctypes.c_int(n_iterations), 
    #             ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Play/Pause
_PlayPause          = _spirit.Simulation_PlayPause
_PlayPause.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, 
                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
_PlayPause.restype  = None
def PlayPause(p_state, method_type, solver_type, n_iterations=-1, n_iterations_log=-1, 
              idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_PlayPause, [ctypes.c_void_p(p_state), 
                                        ctypes.c_char_p(method_type.encode('utf-8')), 
                                        ctypes.c_char_p(solver_type.encode('utf-8')), 
                                        ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log), 
                                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    # _PlayPause(ctypes.c_void_p(p_state), ctypes.c_char_p(method_type), 
    #            ctypes.c_char_p(solver_type), ctypes.c_int(n_iterations), 
    #            ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Stop All
_Stop_All           = _spirit.Simulation_Stop_All
_Stop_All.argtypes  = [ctypes.c_void_p]
_Stop_All.restype   = None
def Stop_All(p_state):
    _Stop_All(ctypes.c_void_p(p_state))

### Check if a simulation is running on a specific image
_Running_Image            = _spirit.Simulation_Running_Image
_Running_Image.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Running_Image.restype    = ctypes.c_bool
def Running_Image(p_state, idx_image=-1, idx_chain=-1):
    return bool(_Running_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                               ctypes.c_int(idx_chain)))

### Check if a simulation is running across a specific chain
_Running_Chain            = _spirit.Simulation_Running_Chain
_Running_Chain.argtypes   = [ctypes.c_void_p, ctypes.c_int]
_Running_Chain.restype    = ctypes.c_bool
def Running_Chain(p_state, idx_chain=-1):
    return bool(_Running_Chain(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Check if a simulation is running on across the collection
_Running_Collection            = _spirit.Simulation_Running_Collection
_Running_Collection.argtypes   = [ctypes.c_void_p]
_Running_Collection.restype    = ctypes.c_bool
def Running_Collection(p_state):
    return bool(_Running_Collection(ctypes.c_void_p(p_state)))


### Check if any simulation running on any image of - or the entire - chain
_Running_Anywhere_Chain           = _spirit.Simulation_Running_Anywhere_Chain
_Running_Anywhere_Chain.argtypes  = [ctypes.c_void_p, ctypes.c_int]
_Running_Anywhere_Chain.restype   = ctypes.c_bool
def Running_Anywhere_Chain(p_state, idx_chain=-1):
    return bool(_Running_Anywhere_Chain(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Check if any simulation running on any image or chain of - or the entire - collection
_Running_Anywhere_Collection            = _spirit.Simulation_Running_Anywhere_Collection
_Running_Anywhere_Collection.argtypes   = [ctypes.c_void_p]
_Running_Anywhere_Collection.restype    = ctypes.c_bool
def Running_Anywhere_Collection(p_state):
    return bool(_Running_Anywhere_Collection(ctypes.c_void_p(p_state)))