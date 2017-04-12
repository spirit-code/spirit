import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Read an image from disk
_Image_Read             = _spirit.IO_Image_Read
_Image_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Image_Read.restype     = None
def Image_Read(p_state, filename, fileformat=0, idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_Image_Read, [p_state, ctypes.c_char_p(filename.encode('utf-8')), fileformat, idx_image, idx_chain])
    # _Image_Read(p_state, ctypes.c_char_p(filename), fileformat, idx_image, idx_chain)

### Write an image to disk
_Image_Write             = _spirit.IO_Image_Write
_Image_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Image_Write.restype     = None
def Image_Write(p_state, filename, fileformat=0, idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_Image_Write, [p_state, ctypes.c_char_p(filename.encode('utf-8')), fileformat, idx_image, idx_chain])
    # _Image_Write(p_state, ctypes.c_char_p(filename), fileformat, idx_image, idx_chain)

### Append an image to an existing file
_Image_Append             = _spirit.IO_Image_Append
_Image_Append.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Image_Append.restype     = None
def Image_Append(p_state, filename, iteration=0, fileformat=0, idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_Image_Append, [p_state, ctypes.c_char_p(filename.encode('utf-8')), iteration, fileformat, idx_image, idx_chain])
    # _Image_Append(p_state, ctypes.c_char_p(filename), iteration, fileformat, idx_image, idx_chain)



### Read a chain of images from disk
_Chain_Read             = _spirit.IO_Chain_Read
_Chain_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Chain_Read.restype     = None
def Chain_Read(p_state, filename, idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_Chain_Read, [p_state, ctypes.c_char_p(filename.encode('utf-8')), idx_image, idx_chain])
    # _Chain_Read(p_state, ctypes.c_char_p(filename), idx_image, idx_chain)

### Write a chain of images to disk
_Chain_Write             = _spirit.IO_Chain_Write
_Chain_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Chain_Write.restype     = None
def Chain_Write(p_state, filename, idx_image=-1, idx_chain=-1):
    spiritlib.WrapFunction(_Chain_Write, [p_state, ctypes.c_char_p(filename.encode('utf-8')), idx_image, idx_chain])
    # _Chain_Write(p_state, ctypes.c_char_p(filename), idx_image, idx_chain)
