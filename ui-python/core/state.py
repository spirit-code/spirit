import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()

### Setup State
_State_Setup = _core.State_Setup
_State_Setup.argtypes = [ctypes.c_char_p]
_State_Setup.restype = ctypes.c_void_p
def setup(configfile):
    return _State_Setup(ctypes.c_char_p(configfile.encode('utf-8')))


### Delete State
_State_Delete = _core.State_Delete
_State_Delete.argtypes = [ctypes.c_void_p]
_State_Delete.restype = None
def delete(p_state):
    return _State_Delete(p_state)