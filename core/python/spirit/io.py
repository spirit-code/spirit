import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### Get the number of images in a file
_N_Images_In_File             = _spirit.IO_N_Images_In_File
_N_Images_In_File.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_N_Images_In_File.restype     = ctypes.c_int
def N_Images_In_File(p_state, filename, idx_image_inchain=-1, idx_chain=-1):
    return int(_N_Images_In_File(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                ctypes.c_int(idx_image_inchain), ctypes.c_int(idx_chain)))

### Read an image from disk
_Image_Read             = _spirit.IO_Image_Read
_Image_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int]
_Image_Read.restype     = None
def Image_Read(p_state, filename, idx_image_infile=0, idx_image_inchain=-1, idx_chain=-1):
    _Image_Read(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                ctypes.c_int(idx_image_infile), ctypes.c_int(idx_image_inchain),
                ctypes.c_int(idx_chain))

### Write an image to disk
_Image_Write             = _spirit.IO_Image_Write
_Image_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                            ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_Image_Write.restype     = None
def Image_Write(p_state, filename, fileformat=6, comment=" ", idx_image=-1, idx_chain=-1):
    _Image_Write(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                 ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                 ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Append an image to an existing file
_Image_Append             = _spirit.IO_Image_Append
_Image_Append.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_Image_Append.restype     = None
def Image_Append(p_state, filename, fileformat=6, comment=" ", idx_image=-1, idx_chain=-1):
    _Image_Append(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                  ctypes.c_int(fileformat), ctypes.c_char_p(filename.encode('utf-8')),
                  ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Read a chain of images from disk
_Chain_Read             = _spirit.IO_Chain_Read
_Chain_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Chain_Read.restype     = None
def Chain_Read(p_state, filename, starting_image=-1, ending_image=-1, insert_idx=-1,
               idx_chain=-1):
    _Chain_Read(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                ctypes.c_int(starting_image), ctypes.c_int(ending_image),
                ctypes.c_int(insert_idx), ctypes.c_int(idx_chain))

### Write a chain of images to disk
_Chain_Write             = _spirit.IO_Chain_Write
_Chain_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                            ctypes.c_int]
_Chain_Write.restype     = None
def Chain_Write(p_state, filename, fileformat=6, comment=" ", idx_chain=-1):
    _Chain_Write(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                 ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                 ctypes.c_int(idx_chain))

### Append a chain of images to disk
_Chain_Append             = _spirit.IO_Chain_Append
_Chain_Append.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                            ctypes.c_int]
_Chain_Append.restype     = None
def Chain_Append(p_state, filename, fileformat=6, comment=" ", idx_chain=-1):
    _Chain_Append(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                 ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                 ctypes.c_int(idx_chain))

