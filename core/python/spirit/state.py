import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### State wrapper class to be used in 'with' statement
class State:
    """Wrapper Class for a Spirit State"""

    def __init__(self, configfile="", quiet=False):
        self.p_state = setup(configfile, quiet)

    def __enter__(self):
        return self.p_state

    def __exit__(self, exc_type, exc_value, traceback):
        delete(self.p_state)


### Setup State
_State_Setup = _spirit.State_Setup
_State_Setup.argtypes = [ctypes.c_char_p, ctypes.c_bool]
_State_Setup.restype = ctypes.c_void_p
def setup(configfile="", quiet=False):
    return _State_Setup(ctypes.c_char_p(configfile.encode('utf-8')), ctypes.c_bool(quiet))

### Delete State
_State_Delete = _spirit.State_Delete
_State_Delete.argtypes = [ctypes.c_void_p]
_State_Delete.restype = None
def delete(p_state):
    return _State_Delete(ctypes.c_void_p(p_state))