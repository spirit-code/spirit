"""
Chain
====================

Manipulate the chain of spin systems (also called images), e.g. add, remove or change active image.
Get information, such as number of images or energies and reaction coordinates.
"""

import spirit.spiritlib as spiritlib
import spirit.parameters as parameters
import spirit.system as system
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()


### Get Chain number of images
_Get_NOI            = _spirit.Chain_Get_NOI
_Get_NOI.argtypes   = [ctypes.c_void_p, ctypes.c_int]
_Get_NOI.restype    = ctypes.c_int
def get_noi(p_state, idx_chain=-1):
    """Get number of images (NOI) in the chain."""
    return int(_Get_NOI(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))


### Switch active to next image of chain
_next_Image             = _spirit.Chain_next_Image
_next_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_next_Image.restype     = ctypes.c_bool
def next_image(p_state, idx_chain=-1):
    """Switch the active image index to the next highest in the chain."""
    return bool(_next_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Switch active to previous image of chain
_prev_Image             = _spirit.Chain_prev_Image
_prev_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_prev_Image.restype     = ctypes.c_bool
def prev_image(p_state, idx_chain=-1):
    """Switch the active image index to the next lowest in the chain."""
    return bool(_prev_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Switch active to specific image of chain
_Jump_To_Image             = _spirit.Chain_Jump_To_Image
_Jump_To_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Jump_To_Image.restype     = ctypes.c_bool
def jump_to_image(p_state, idx_image=-1, idx_chain=-1):
    """Set the index of the active image in the chain."""
    return bool(_Jump_To_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))


### Set the number of images in the chain
_Set_Length             = _spirit.Chain_Set_Length
_Set_Length.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Set_Length.restype     = None
def set_length(p_state, n_images, idx_chain=-1):
    """Set the number of images (NOI) in the chain.

    If the chain is longer, the corresponding number of images is erased from the end.

    If the chain is shorter, the corresponding number of images is appended.

    Note that the active image might change.

    If no image is in the clipboard, no action is taken.
    """
    _Set_Length(ctypes.c_void_p(p_state), ctypes.c_int(n_images), ctypes.c_int(idx_chain))

### Copy active image to clipboard
_Image_to_Clipboard             = _spirit.Chain_Image_to_Clipboard
_Image_to_Clipboard.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Image_to_Clipboard.restype     = None
def image_to_clipboard(p_state, idx_image=-1, idx_chain=-1):
    """Copies an image to the clipboard of Spirit. It can then be later e.g. inserted or appended."""
    _Image_to_Clipboard(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Replace active image in chain
_Replace_Image             = _spirit.Chain_Replace_Image
_Replace_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Replace_Image.restype     = None
def replace_image(p_state, idx_image=-1, idx_chain=-1):
    """Replaces the image from the one in the clipboard.

    If no image is in the clipboard, no action is taken.
    """
    _Replace_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### Insert clipboard image before image in chain
_Insert_Image_Before             = _spirit.Chain_Insert_Image_Before
_Insert_Image_Before.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Insert_Image_Before.restype     = None
def insert_image_before(p_state, idx_image=-1, idx_chain=-1):
    """Inserts an image in front of the specified index.

    Note that the active image might change.

    If no image is in the clipboard, no action is taken.
    """
    _Insert_Image_Before(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Insert clipboard image after image in chain
_Insert_Image_After             = _spirit.Chain_Insert_Image_After
_Insert_Image_After.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Insert_Image_After.restype     = None
def insert_image_after(p_state, idx_image=-1, idx_chain=-1):
    """Inserts an image after the specified index.

    Note that the active image might change.

    If no image is in the clipboard, no action is taken.
    """
    _Insert_Image_After(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Insert clipboard image at end of chain
_Push_Back             = _spirit.Chain_Push_Back
_Push_Back.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Push_Back.restype     = None
def push_back(p_state, idx_chain=-1):
    """Appends an image to the chain.

    If no image is in the clipboard, no action is taken.
    """
    _Push_Back(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))



### Delete active image
_Delete_Image             = _spirit.Chain_Delete_Image
_Delete_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Delete_Image.restype     = ctypes.c_bool
def delete_image(p_state, idx_image=-1, idx_chain=-1):
    """Removes the specified image from the chain.

    Note that the active image might change.

    If it is the last remaining image in the chain, no action is taken.
    """
    return bool(_Delete_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Delete image at end of chain
_Pop_Back             = _spirit.Chain_Pop_Back
_Pop_Back.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Pop_Back.restype     = ctypes.c_bool
def pop_back(p_state, idx_chain=-1):
    """Removes the last image from the chain.

    Note that the active image might change.

    If it is the last remaining image in the chain, no action is taken.
    """
    return bool(_Pop_Back(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))


### Update the chain's data (interpolated energies etc.)
_Update_Data             = _spirit.Chain_Update_Data
_Update_Data.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Update_Data.restype     = None
def update_data(p_state, idx_chain=-1):
    """Updates various data of the chain, including:

    - Energies of images
    - Reaction coordinates of images
    - Interpolated energy and reaction coordinate values
    """
    _Update_Data(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))


### Setup the chain's data arrays (when is this necessary?)
_Setup_Data             = _spirit.Chain_Setup_Data
_Setup_Data.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Setup_Data.restype     = None
def setup_data(p_state, idx_chain=-1):
    _Setup_Data(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))


### Get Rx
_Get_Rx          = _spirit.Chain_Get_Rx
_Get_Rx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Rx.restype  = None
def get_reaction_coordinate(p_state, idx_chain=-1):
    """Returns an array of shape (NOI) containing the reaction coordinates of the images."""
    noi = get_noi(p_state, idx_chain)
    Rx = (noi*ctypes.c_float)()
    _Get_Rx(ctypes.c_void_p(p_state), Rx, ctypes.c_int(idx_chain))
    return [x for x in Rx]

### Get Rx interpolated
_Get_Rx_Interpolated          = _spirit.Chain_Get_Rx_Interpolated
_Get_Rx_Interpolated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Rx_Interpolated.restype  = None
def get_reaction_coordinate_interpolated(p_state, idx_chain=-1):
    """Returns an array containing the interpolated reaction coordinate values along the chain.

    The number of interpolated values between images can be set in the GNEB parameters.
    """
    noi = get_noi(p_state, idx_chain)
    n_interp = parameters.gneb.get_n_energy_interpolations(p_state, idx_chain)
    len_Rx = noi + (noi-1)*n_interp
    Rx = (len_Rx*ctypes.c_float)()
    _Get_Rx_Interpolated(ctypes.c_void_p(p_state), Rx, ctypes.c_int(idx_chain))
    return [x for x in Rx]

### Get Energy
_Get_Energy          = _spirit.Chain_Get_Energy
_Get_Energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Energy.restype  = None
def get_energy(p_state, idx_chain=-1):
    """Returns an array of shape (NOI) containing the energies of the images."""
    noi = get_noi(p_state, idx_chain)
    Energy = (noi*ctypes.c_float)()
    _Get_Energy(ctypes.c_void_p(p_state), Energy, ctypes.c_int(idx_chain))
    return [E for E in Energy]

### Get Energy Interpolated
_Get_Energy_Interpolated          = _spirit.Chain_Get_Energy_Interpolated
_Get_Energy_Interpolated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Energy_Interpolated.restype  = None
def get_energy_interpolated(p_state, idx_chain=-1):
    """Returns an array containing the interpolated energy values along the chain.

    The number of interpolated values between images can be set in the GNEB parameters.
    """
    noi = get_noi(p_state, idx_chain)
    n_interp = parameters.gneb.get_n_energy_interpolations(p_state, idx_chain)
    len_Energy = noi + (noi-1)*n_interp
    Energy_interp = (len_Energy*ctypes.c_float)()
    _Get_Energy_Interpolated(ctypes.c_void_p(p_state), Energy_interp, ctypes.c_int(idx_chain))
    return [E for E in Energy_interp]