import corelib
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


### Get Chain index
_Get_Index          = _core.Chain_Get_Index
_Get_Index.argtypes = [ctypes.c_void_p]
_Get_Index.restype  = ctypes.c_int
def Get_Index(p_state):
    return int(_Get_Index(p_state))


### Get Chain number of images
_Get_NOI            = _core.Chain_Get_NOI
_Get_NOI.argtypes   = [ctypes.c_void_p]
_Get_NOI.restype    = ctypes.c_int
def Get_NOI(p_state):
    return int(_Get_NOI(p_state))


### Switch active to next image of chain
_next_Image             = _core.Chain_next_Image
_next_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_next_Image.restype     = None
def Next_Image(p_state, idx_chain=-1):
    _next_Image(p_state, idx_chain)


### Switch active to next image of chain
_prev_Image             = _core.Chain_prev_Image
_prev_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_prev_Image.restype     = None
def Prev_Image(p_state, idx_chain=-1):
    _prev_Image(p_state, idx_chain)


### Copy active image to clipboard
_Image_to_Clipboard             = _core.Chain_Image_to_Clipboard
_Image_to_Clipboard.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Image_to_Clipboard.restype     = None
def Image_to_Clipboard(p_state, idx_image=-1, idx_chain=-1):
    _Image_to_Clipboard(p_state, idx_image, idx_chain)


### Insert clipboard image before active in chain
_Insert_Image_Before             = _core.Chain_Insert_Image_Before
_Insert_Image_Before.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Insert_Image_Before.restype     = None
def Insert_Image_Before(p_state, idx_image=-1, idx_chain=-1):
    _Insert_Image_Before(p_state, idx_image, idx_chain)


### Insert clipboard image before active in chain
_Insert_Image_After             = _core.Chain_Insert_Image_After
_Insert_Image_After.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Insert_Image_After.restype     = None
def Insert_Image_After(p_state, idx_image=-1, idx_chain=-1):
    _Insert_Image_After(p_state, idx_image, idx_chain)


### Insert clipboard image before active in chain
_Replace_Image             = _core.Chain_Replace_Image
_Replace_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Replace_Image.restype     = None
def Replace_Image(p_state, idx_image=-1, idx_chain=-1):
    _Replace_Image(p_state, idx_image, idx_chain)


### Insert clipboard image before active in chain
_Delete_Image             = _core.Chain_Delete_Image
_Delete_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Delete_Image.restype     = None
def Delete_Image(p_state, idx_image=-1, idx_chain=-1):
    _Delete_Image(p_state, idx_image, idx_chain)


### Insert clipboard image before active in chain
_Delete_Image             = _core.Chain_Delete_Image
_Delete_Image.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Delete_Image.restype     = None
def Delete_Image(p_state, idx_image=-1, idx_chain=-1):
    _Delete_Image(p_state, idx_image, idx_chain)


### Insert clipboard image before active in chain
_Update_Data             = _core.Chain_Update_Data
_Update_Data.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Update_Data.restype     = None
def Update_Data(p_state, idx_chain=-1):
    _Update_Data(p_state, idx_chain)


### Insert clipboard image before active in chain
_Setup_Data             = _core.Chain_Setup_Data
_Setup_Data.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_Setup_Data.restype     = None
def Setup_Data(p_state, idx_chain=-1):
    _Setup_Data(p_state, idx_chain)