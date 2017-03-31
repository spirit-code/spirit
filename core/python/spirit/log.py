import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Send a Log message
_Send             = _spirit.Log_Send
_Send.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_Send.restype     = None
def Send(p_state, level, sender, message, idx_image=-1, idx_chain=-1):
    _Send(p_state, level, sender, ctypes.c_char_p(message.encode('utf-8')), idx_image, idx_chain)

### Append Log to file
_Append             = _spirit.Log_Append
_Append.argtypes    = [ctypes.c_void_p]
_Append.restype     = None
def Append(p_state):
    _Append(p_state)