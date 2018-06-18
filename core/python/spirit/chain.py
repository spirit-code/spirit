import spirit.spiritlib as spiritlib
import spirit.parameters as parameters
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Get Chain index
_Get_Index          = _spirit.Chain_Get_Index
_Get_Index.argtypes = [ctypes.c_void_p]
_Get_Index.restype  = ctypes.c_int
def Get_Index(p_state):
    return int(_Get_Index(ctypes.c_void_p(p_state)))


### Get Chain number of images
_Get_NOI            = _spirit.Chain_Get_NOI
_Get_NOI.argtypes   = [ctypes.c_void_p, ctypes.c_int]
_Get_NOI.restype    = ctypes.c_int
def Get_NOI(p_state, idx_chain=-1):
    return int(_Get_NOI(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))


### Switch active to next image of chain
_next_Image             = _spirit.Chain_next_Image
_next_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_next_Image.restype     = ctypes.c_bool
def Next_Image(p_state, idx_chain=-1):
    return bool(_next_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Switch active to previous image of chain
_prev_Image             = _spirit.Chain_prev_Image
_prev_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_prev_Image.restype     = ctypes.c_bool
def Prev_Image(p_state, idx_chain=-1):
    return bool(_prev_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))

### Switch active to specific image of chain
_Jump_To_Image             = _spirit.Chain_Jump_To_Image
_Jump_To_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Jump_To_Image.restype     = ctypes.c_bool
def Jump_To_Image(p_state, idx_image=-1, idx_chain=-1):
    return bool(_Jump_To_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))


### Copy active image to clipboard
_Image_to_Clipboard             = _spirit.Chain_Image_to_Clipboard
_Image_to_Clipboard.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Image_to_Clipboard.restype     = None
def Image_to_Clipboard(p_state, idx_image=-1, idx_chain=-1):
    _Image_to_Clipboard(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Replace active image in chain
_Replace_Image             = _spirit.Chain_Replace_Image
_Replace_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Replace_Image.restype     = None
def Replace_Image(p_state, idx_image=-1, idx_chain=-1):
    _Replace_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


### Insert clipboard image before image in chain
_Insert_Image_Before             = _spirit.Chain_Insert_Image_Before
_Insert_Image_Before.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Insert_Image_Before.restype     = None
def Insert_Image_Before(p_state, idx_image=-1, idx_chain=-1):
    _Insert_Image_Before(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Insert clipboard image after image in chain
_Insert_Image_After             = _spirit.Chain_Insert_Image_After
_Insert_Image_After.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Insert_Image_After.restype     = None
def Insert_Image_After(p_state, idx_image=-1, idx_chain=-1):
    _Insert_Image_After(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Insert clipboard image at end of chain
_Push_Back             = _spirit.Chain_Push_Back
_Push_Back.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Push_Back.restype     = None
def Push_Back(p_state, idx_chain=-1):
    _Push_Back(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))



### Delete active image
_Delete_Image             = _spirit.Chain_Delete_Image
_Delete_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Delete_Image.restype     = ctypes.c_bool
def Delete_Image(p_state, idx_image=-1, idx_chain=-1):
    return bool(_Delete_Image(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Delete image at end of chain
_Pop_Back             = _spirit.Chain_Pop_Back
_Pop_Back.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Pop_Back.restype     = ctypes.c_bool
def Pop_Back(p_state, idx_chain=-1):
    return bool(_Pop_Back(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))


### Update the chain's data (interpolated energies etc.)
_Update_Data             = _spirit.Chain_Update_Data
_Update_Data.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Update_Data.restype     = None
def Update_Data(p_state, idx_chain=-1):
    _Update_Data(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))


### Setup the chain's data arrays (when is this necessary?)
_Setup_Data             = _spirit.Chain_Setup_Data
_Setup_Data.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Setup_Data.restype     = None
def Setup_Data(p_state, idx_chain=-1):
    _Setup_Data(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))


### Get Rx
_Get_Rx          = _spirit.Chain_Get_Rx
_Get_Rx.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Rx.restype  = None
def Get_Rx(p_state, idx_chain=-1):
    noi = Get_NOI(p_state, idx_chain)
    Rx = (noi*ctypes.c_float)()
    _Get_Rx(ctypes.c_void_p(p_state), Rx, ctypes.c_int(idx_chain))
    return [x for x in Rx]

### Get Rx interpolated
_Get_Rx_Interpolated          = _spirit.Chain_Get_Rx_Interpolated
_Get_Rx_Interpolated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Rx_Interpolated.restype  = None
def Get_Rx_Interpolated(p_state, idx_chain=-1):
    noi = Get_NOI(p_state, idx_chain)
    n_interp = parameters.gneb.getEnergyInterpolations(p_state, idx_chain)
    len_Rx = noi + (noi-1)*n_interp
    Rx = (len_Rx*ctypes.c_float)()
    _Get_Rx_Interpolated(ctypes.c_void_p(p_state), Rx, ctypes.c_int(idx_chain))
    return [x for x in Rx]

### Get Energy
_Get_Energy          = _spirit.Chain_Get_Energy
_Get_Energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Energy.restype  = None
def Get_Energy(p_state, idx_chain=-1):
    noi = Get_NOI(p_state, idx_chain)
    Energy = (noi*ctypes.c_float)()
    _Get_Energy(ctypes.c_void_p(p_state), Energy, ctypes.c_int(idx_chain))
    return [E for E in Energy]

### Get Energy Interpolated
_Get_Energy_Interpolated          = _spirit.Chain_Get_Energy_Interpolated
_Get_Energy_Interpolated.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Energy_Interpolated.restype  = None
def Get_Energy_Interpolated(p_state, idx_chain=-1):
    noi = Get_NOI(p_state, idx_chain)
    n_interp = parameters.gneb.getEnergyInterpolations(p_state, idx_chain)
    len_Energy = noi + (noi-1)*n_interp
    Energy_interp = (len_Energy*ctypes.c_float)()
    _Get_Energy_Interpolated(ctypes.c_void_p(p_state), Energy_interp, ctypes.c_int(idx_chain))
    return [E for E in Energy_interp]