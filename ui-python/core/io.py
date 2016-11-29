import core.corelib as corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Domain Wall configuration
_Image_Read             = _core.IO_Image_Read
_Image_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Image_Read.restype     = None
def Image_Read(p_state, filename, fileformat=0, idx_image=-1, idx_chain=-1):
    _Image_Read(p_state, ctypes.c_char_p(filename), fileformat, idx_image, idx_chain)

_Image_Write             = _core.IO_Image_Write
_Image_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Image_Write.restype     = None
def Image_Write(p_state, filename, fileformat=0, idx_image=-1, idx_chain=-1):
    _Image_Write(p_state, ctypes.c_char_p(filename), fileformat, idx_image, idx_chain)

_Image_Append             = _core.IO_Image_Append
_Image_Append.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Image_Append.restype     = None
def Image_Append(p_state, filename, iteration=0, fileformat=0, idx_image=-1, idx_chain=-1):
    _Image_Append(p_state, ctypes.c_char_p(filename), iteration, fileformat, idx_image, idx_chain)