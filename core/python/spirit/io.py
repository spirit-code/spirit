"""
I/O
====================

Read and write spin configurations, chains or eigenmodes.
Vectorfields are generally written in the `OOMMF vector field (OVF) file format <http://math.nist.gov/oommf/doc/userguide12a5/userguide/OVF_2.0_format.html>`_.

Note that, when reading an image or chain from file, the file will automatically be tested for an OVF header.
If it cannot be identified as OVF, it will be tried to be read as three plain text columns (Sx Sy Sz).

Note also, IO is still being re-written and only OVF will be supported as output format.
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

### Output file formats
FILEFORMAT_OVF_BIN  = 0
"""OVF binary format corresponding to the precision with which the code was compiled"""

FILEFORMAT_OVF_BIN4 = 1
"""OVF binary format with precision 4"""

FILEFORMAT_OVF_BIN8 = 2
"""OVF binary format with precision 8"""

FILEFORMAT_OVF_TEXT = 3
"""OVF text format"""

FILEFORMAT_OVF_CSV  = 4
"""OVF text format with comma-separated columns"""

_N_Images_In_File             = _spirit.IO_N_Images_In_File
_N_Images_In_File.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_N_Images_In_File.restype     = ctypes.c_int
def n_images_in_file(p_state, filename, idx_image_inchain=-1, idx_chain=-1):
    """Returns the number of segments or images in a given file.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to check
    """
    return int(_N_Images_In_File(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                ctypes.c_int(idx_image_inchain), ctypes.c_int(idx_chain)))

_Image_Read             = _spirit.IO_Image_Read
_Image_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int]
_Image_Read.restype     = None
def image_read(p_state, filename, idx_image_infile=0, idx_image_inchain=-1, idx_chain=-1):
    """Attempt to read a spin configuration from a file into an image of the chain.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to read

    Keyword arguments:
    idx_image_infile -- the index of the image in the file which should be read in (default: 0)
    idx_image_inchain -- the index of the image in the chain into which the data should be read (default: active image)
    """
    _Image_Read(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                ctypes.c_int(idx_image_infile), ctypes.c_int(idx_image_inchain),
                ctypes.c_int(idx_chain))

_Image_Write             = _spirit.IO_Image_Write
_Image_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                            ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_Image_Write.restype     = None
def image_write(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment="", idx_image=-1, idx_chain=-1):
    """Write an image of the chain to a file.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to write

    Keyword arguments:
    fileformat -- the format in which to write the data (default: OVF text)
    comment -- a comment string to be inserted in the header (default: empty)
    idx_image -- the index of the image to be written to the file (default: active image)
    """
    _Image_Write(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                 ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                 ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Image_Append             = _spirit.IO_Image_Append
_Image_Append.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                             ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_Image_Append.restype     = None
def image_append(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment="", idx_image=-1, idx_chain=-1):
    """Append an image of the chain to a file, incrementing the segment count.

    If the file does not exist, it is created.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to append to

    Keyword arguments:
    fileformat -- the format in which to write the data (default: OVF text)
    comment -- a comment string to be inserted in the header (default: empty)
    idx_image -- the index of the image to be written to the file (default: active image)
    """
    _Image_Append(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                  ctypes.c_int(fileformat), ctypes.c_char_p(filename.encode('utf-8')),
                  ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Chain_Read             = _spirit.IO_Chain_Read
_Chain_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Chain_Read.restype     = None
def chain_read(p_state, filename, starting_image=0, ending_image=0, insert_idx=-1, idx_chain=-1):
    """Attempt to read a chain of images from a given file.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to read

    Keyword arguments:
    starting_image -- the index within the file at which to start reading (default: 0)
    ending_image -- the index within the file at which to stop reading (default: 0)
    insert_idx -- the index within the chain at which to start placing the images (default: active image)

    Images of the chain will be overwritten with what is read from the file.
    If the chain is not long enough for the number of images to be read, it is automatically set to the
    right length.
    """
    _Chain_Read(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                ctypes.c_int(starting_image), ctypes.c_int(ending_image),
                ctypes.c_int(insert_idx), ctypes.c_int(idx_chain))

_Chain_Write             = _spirit.IO_Chain_Write
_Chain_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                            ctypes.c_int]
_Chain_Write.restype     = None
def chain_write(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment="", idx_chain=-1):
    """Write a chain of images to a file.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to write

    Keyword arguments:
    fileformat -- the format in which to write the data (default: OVF text)
    comment -- a comment string to be inserted in the header (default: empty)
    """
    _Chain_Write(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                 ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                 ctypes.c_int(idx_chain))

_Chain_Append             = _spirit.IO_Chain_Append
_Chain_Append.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                            ctypes.c_int]
_Chain_Append.restype     = None
def chain_append(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment="", idx_chain=-1):
    """Append a chain of images to a given file.

    If the file does not exist, it is created.

    Arguments:
    p_state -- state pointer
    filename -- the name of the file to append to

    Keyword arguments:
    fileformat -- the format in which to write the data (default: OVF text)
    comment -- a comment string to be inserted in the header (default: empty)
    """
    _Chain_Append(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                 ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                 ctypes.c_int(idx_chain))

_Eigenmodes_Read             = _spirit.IO_Eigenmodes_Read
_Eigenmodes_Read.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int]
_Eigenmodes_Read.restype     = None
def eigenmodes_read(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, idx_image_inchain=-1, idx_chain=-1):
    """Read the vector fields from a file as a set of eigenmodes for the spin system."""
    _Eigenmodes_Read(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                     ctypes.c_int(fileformat), ctypes.c_int(idx_image_inchain),
                     ctypes.c_int(idx_chain))

_Eigenmodes_Write             = _spirit.IO_Eigenmodes_Write
_Eigenmodes_Write.argtypes    = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                 ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_Eigenmodes_Write.restype     = None
def eigenmodes_write(p_state, filename, fileformat=FILEFORMAT_OVF_TEXT, comment=" ", idx_image=-1, idx_chain=-1):
    """Write the eigenmodes of a spin system to file, if they have been already calculated."""
    _Eigenmodes_Write(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')),
                      ctypes.c_int(fileformat), ctypes.c_char_p(comment.encode('utf-8')),
                      ctypes.c_int(idx_image), ctypes.c_int(idx_chain))