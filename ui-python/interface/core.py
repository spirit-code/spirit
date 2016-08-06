import os 
import ctypes

### Get this file's directory. The library should be here
core_dir = os.path.dirname(os.path.realpath(__file__))
### Load the core library
_core = ctypes.CDLL(core_dir + '/libcore.so')

### Setup State
_setupState = _core.setupState
_setupState.argtypes = [ctypes.c_char_p]
_setupState.restype = ctypes.c_void_p
def setupState(configfile):
    global _core
    # n = len(configfile)
    # array_type = ctypes.c_char_p * n
    # result = _core.setupState(ctypes.c_char_p(configfile))
    # return state(result)
    return _core.setupState(ctypes.c_char_p(configfile))