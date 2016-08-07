import corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()

### Setup State
_setupState = _core.setupState
_setupState.argtypes = [ctypes.c_char_p]
_setupState.restype = ctypes.c_void_p
def setup(configfile):
    global _core
    return _core.setupState(ctypes.c_char_p(configfile))