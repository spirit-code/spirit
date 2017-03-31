import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### State wrapper class to be used in 'with' statement
class State:
    """Wrapper Class for a Spirit State"""

    def __init__(self, configfile):
        self.p_state = setup(configfile)

    def __enter__(self):
        return self.p_state

    def __exit__(self, exc_type, exc_value, traceback):
        delete(self.p_state)


### Setup State
_State_Setup = _spirit.State_Setup
_State_Setup.argtypes = [ctypes.c_char_p]
_State_Setup.restype = ctypes.c_void_p
def setup(configfile):
    return _State_Setup(ctypes.c_char_p(configfile.encode('utf-8')))

### Delete State
_State_Delete = _spirit.State_Delete
_State_Delete.argtypes = [ctypes.c_void_p]
_State_Delete.restype = None
def delete(p_state):
    return _State_Delete(p_state)