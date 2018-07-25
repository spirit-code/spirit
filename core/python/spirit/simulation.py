import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

import threading

###     We use a thread for PlayPause, so that KeyboardInterrupt can be forwarded to the CDLL call
###     We might want to think about using PyDLL and about a signal handler in the core library
###     see here: http://stackoverflow.com/questions/14271697/ctrlc-doesnt-interrupt-call-to-shared-library-using-ctypes-in-python

SOLVER_VP = 0
SOLVER_SIB = 1
SOLVER_DEPONDT = 2
SOLVER_HEUN = 3

METHOD_MC   = 0
METHOD_LLG  = 1
METHOD_GNEB = 2
METHOD_MMF  = 3
METHOD_EMA  = 4

### ----- Start
### MC
_MC_Start          = _spirit.Simulation_MC_Start
_MC_Start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                        ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MC_Start.restype  = None
### LLG
_LLG_Start          = _spirit.Simulation_LLG_Start
_LLG_Start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_LLG_Start.restype  = None
### GNEB
_GNEB_Start          = _spirit.Simulation_GNEB_Start
_GNEB_Start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_bool, ctypes.c_int]
_GNEB_Start.restype  = None
### MMF
_MMF_Start          = _spirit.Simulation_MMF_Start
_MMF_Start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MMF_Start.restype  = None
### EMA
_EMA_Start          = _spirit.Simulation_EMA_Start
_EMA_Start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                        ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_EMA_Start.restype  = None
### Wrapper
def start(p_state, method_type, solver_type=None, n_iterations=-1, n_iterations_log=-1,
            single_shot=False, idx_image=-1, idx_chain=-1):

    if method_type == METHOD_MC:
        spiritlib.wrap_function(_MC_Start, [ctypes.c_void_p(p_state),
                                            ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log),
                                            ctypes.c_bool(single_shot),
                                            ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    elif method_type == METHOD_LLG:
        spiritlib.wrap_function(_LLG_Start, [ctypes.c_void_p(p_state),
                                            ctypes.c_int(solver_type),
                                            ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log),
                                            ctypes.c_bool(single_shot),
                                            ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    elif method_type == METHOD_GNEB:
        spiritlib.wrap_function(_GNEB_Start, [ctypes.c_void_p(p_state),
                                            ctypes.c_int(solver_type),
                                            ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log),
                                            ctypes.c_bool(single_shot),
                                            ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    elif method_type == METHOD_MMF:
        spiritlib.wrap_function(_MMF_Start, [ctypes.c_void_p(p_state),
                                            ctypes.c_int(solver_type),
                                            ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log),
                                            ctypes.c_bool(single_shot),
                                            ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    elif method_type == METHOD_EMA:
        spiritlib.wrap_function(_EMA_Start, [ctypes.c_void_p(p_state),
                                            ctypes.c_int(n_iterations), ctypes.c_int(n_iterations_log),
                                            ctypes.c_bool(single_shot),
                                            ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    else:
        print("Invalid method_type passed to simulation.start...")

    # _Start(ctypes.c_void_p(p_state), ctypes.c_char_p(method_type),
    #            ctypes.c_char_p(solver_type), ctypes.c_int(n_iterations),
    #            ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### SingleShot Iteration
_SingleShot          = _spirit.Simulation_SingleShot
_SingleShot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_SingleShot.restype  = None
def single_shot(p_state, idx_image=-1, idx_chain=-1):
    spiritlib.wrap_function(_SingleShot, [ctypes.c_void_p(p_state),
                                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain)])
    # _SingleShot(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Stop current
_Stop           = _spirit.Simulation_Stop
_Stop.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Stop.restype   = None
def stop(p_state, idx_image=-1, idx_chain=-1):
    _Stop(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Stop all
_Stop_All           = _spirit.Simulation_Stop_All
_Stop_All.argtypes  = [ctypes.c_void_p]
_Stop_All.restype   = None
def stop_all(p_state):
    _Stop_All(ctypes.c_void_p(p_state))

### Check if a simulation is running on a specific image
_Running_On_Image            = _spirit.Simulation_Running_On_Image
_Running_On_Image.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Running_On_Image.restype    = ctypes.c_bool
def running_on_image(p_state, idx_image=-1, idx_chain=-1):
    return bool(_Running_On_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                               ctypes.c_int(idx_chain)))

### Check if a simulation is running across a specific chain
_Running_On_Chain            = _spirit.Simulation_Running_On_Chain
_Running_On_Chain.argtypes   = [ctypes.c_void_p, ctypes.c_int]
_Running_On_Chain.restype    = ctypes.c_bool
def running_on_chain(p_state, idx_chain=-1):
    return bool(_Running_On_Chain(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Check if any simulation running on any image of - or the entire - chain
_Running_Anywhere_On_Chain           = _spirit.Simulation_Running_Anywhere_On_Chain
_Running_Anywhere_On_Chain.argtypes  = [ctypes.c_void_p, ctypes.c_int]
_Running_Anywhere_On_Chain.restype   = ctypes.c_bool
def running_anywhere_on_chain(p_state, idx_chain=-1):
    return bool(_Running_Anywhere_On_Chain(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))