import spirit.spiritlib as spiritlib
from spirit.io import FILEFORMAT_OVF_TEXT
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

### ---------------------------------- Set ----------------------------------

### Set the output file tag, which is placed in front of all output files of GNEB simulations
_GNEB_Set_Output_Tag          = _spirit.Parameters_GNEB_Set_Output_Tag
_GNEB_Set_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Tag.restype  = None
def set_output_tag(p_state, tag, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output folder for GNEB simulations
_GNEB_Set_Output_Folder          = _spirit.Parameters_GNEB_Set_Output_Folder
_GNEB_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Folder.restype  = None
def set_output_folder(p_state, folder, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration
_GNEB_Set_Output_General          = _spirit.Parameters_GNEB_Set_Output_General
_GNEB_Set_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_General.restype  = None
def set_output_general(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration for energy files
_GNEB_Set_Output_Energies          = _spirit.Parameters_GNEB_Set_Output_Energies
_GNEB_Set_Output_Energies.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Energies.restype  = None
def set_output_energies(p_state, step=True, interpolated=True, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Output_Energies(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(interpolated),
                        ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration for energy files
_GNEB_Set_Output_Chain          = _spirit.Parameters_GNEB_Set_Output_Chain
_GNEB_Set_Output_Chain.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Chain.restype  = None
def set_output_configuration(p_state, step=False, filetype=FILEFORMAT_OVF_TEXT, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Output_Chain(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_int(filetype),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB N Iterations
_GNEB_Set_N_Iterations             = _spirit.Parameters_GNEB_Set_N_Iterations
_GNEB_Set_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int]
_GNEB_Set_N_Iterations.restype     = None
def set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _GNEB_Set_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                           ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                           ctypes.c_int(idx_chain))

### Set GNEB convergence
_GNEB_Set_Convergence           = _spirit.Parameters_GNEB_Set_Convergence
_GNEB_Set_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Convergence.restype   = None
def set_convergence(p_state, convergence, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Convergence(ctypes.c_void_p(p_state), ctypes.c_float(convergence),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB Spring Constant
_GNEB_Set_Spring_Constant           = _spirit.Parameters_GNEB_Set_Spring_Constant
_GNEB_Set_Spring_Constant.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Spring_Constant.restype   = None
def set_spring_constant(p_state, c_spring, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Spring_Constant(ctypes.c_void_p(p_state), ctypes.c_float(c_spring),
                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB climbing and falling images
_GNEB_Set_Climbing_Falling             = _spirit.Parameters_GNEB_Set_Climbing_Falling
_GNEB_Set_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Climbing_Falling.restype     = None
def set_climbing_falling(p_state, image_type, idx_image=-1, idx_chain=-1):
    _GNEB_Set_Climbing_Falling(ctypes.c_void_p(p_state), ctypes.c_int(image_type),
                               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set GNEB climbing and falling images
_GNEB_Set_Image_Type_Automatically             = _spirit.Parameters_GNEB_Set_Image_Type_Automatically
_GNEB_Set_Image_Type_Automatically.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_GNEB_Set_Image_Type_Automatically.restype     = None
def set_image_type_automatically(p_state, idx_chain=-1):
    _GNEB_Set_Image_Type_Automatically(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))

### ---------------------------------- Get ----------------------------------

### Get GNEB N Iterations
_GNEB_Get_N_Iterations             = _spirit.Parameters_GNEB_Get_N_Iterations
_GNEB_Get_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                      ctypes.POINTER( ctypes.c_int ), ctypes.c_int]
_GNEB_Get_N_Iterations.restype     = None
def get_iterations(p_state, idx_image=-1, idx_chain=-1):
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _GNEB_Get_N_Iterations(ctypes.c_void_p(p_state), ctypes.pointer(n_iterations),
                           ctypes.pointer(n_iterations_log), ctypes.c_int(idx_image),
                           ctypes.c_int(idx_chain) )
    return int(n_iterations.value), int(n_iterations_log.value)

### Get GNEB convergence
_GNEB_Get_Convergence           = _spirit.Parameters_GNEB_Get_Convergence
_GNEB_Get_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Convergence.restype   = ctypes.c_float
def get_convergence(p_state, idx_image=-1, idx_chain=-1):
    return float( _GNEB_Get_Convergence(ctypes.c_void_p(p_state), 
                                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB Spring Constant
_GNEB_Get_Spring_Constant             = _spirit.Parameters_GNEB_Get_Spring_Constant
_GNEB_Get_Spring_Constant.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Spring_Constant.restype     = ctypes.c_float
def get_spring_constant(p_state, idx_image=-1, idx_chain=-1):
    return float(_GNEB_Get_Spring_Constant(ctypes.c_void_p(p_state), 
                                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB climbing and falling images
_GNEB_Get_Climbing_Falling             = _spirit.Parameters_GNEB_Get_Climbing_Falling
_GNEB_Get_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Climbing_Falling.restype     = ctypes.c_int
def get_climbing_falling(p_state, idx_image=-1, idx_chain=-1):
    return int(_GNEB_Get_Climbing_Falling(ctypes.c_void_p(p_state), 
                                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get GNEB number of energy interpolations
_GNEB_Get_N_Energy_Interpolations             = _spirit.Parameters_GNEB_Get_N_Energy_Interpolations
_GNEB_Get_N_Energy_Interpolations.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_GNEB_Get_N_Energy_Interpolations.restype     = ctypes.c_int
def get_n_energy_interpolations(p_state, idx_chain=-1):
    return int(_GNEB_Get_N_Energy_Interpolations(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))