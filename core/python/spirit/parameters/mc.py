import spirit.spiritlib as spiritlib
from spirit.io import FILEFORMAT_OVF_TEXT
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

## ---------------------------------- Set ----------------------------------

### Set the output file tag, which is placed in front of all output files of MC simulations
_MC_Set_Output_Tag          = _spirit.Parameters_MC_Set_Output_Tag
_MC_Set_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Tag.restype  = None
def set_output_tag(p_state, tag, idx_image=-1, idx_chain=-1):
    _MC_Set_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output folder for MC simulations
_MC_Set_Output_Folder          = _spirit.Parameters_MC_Set_Output_Folder
_MC_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Folder.restype  = None
def set_output_folder(p_state, folder, idx_image=-1, idx_chain=-1):
    _MC_Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration
_MC_Set_Output_General          = _spirit.Parameters_MC_Set_Output_General
_MC_Set_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MC_Set_Output_General.restype  = None
def set_output_general(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    _MC_Set_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration for energy files
_MC_Set_Output_Energy          = _spirit.Parameters_MC_Set_Output_Energy
_MC_Set_Output_Energy.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Energy.restype  = None
def set_output_energy(p_state, step=False, archive=True, spin_resolved=False, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    _MC_Set_Output_Energy(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_bool(spin_resolved), ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration for energy files
_MC_Set_Output_Configuration          = _spirit.Parameters_MC_Set_Output_Configuration
_MC_Set_Output_Configuration.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Configuration.restype  = None
def set_output_configuration(p_state, step=False, archive=True, filetype=FILEFORMAT_OVF_TEXT, idx_image=-1, idx_chain=-1):
    _MC_Set_Output_Configuration(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_int(filetype), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set number of iterations and step size
_MC_Set_N_Iterations             = _spirit.Parameters_MC_Set_N_Iterations
_MC_Set_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int]
_MC_Set_N_Iterations.restype     = None
def set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _MC_Set_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                          ctypes.c_int(idx_chain))

### Set temperature
_MC_Set_Temperature             = _spirit.Parameters_MC_Set_Temperature
_MC_Set_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_MC_Set_Temperature.restype     = None
def set_temperature(p_state, temperature, idx_image=-1, idx_chain=-1):
    _MC_Set_Temperature(ctypes.c_void_p(p_state), ctypes.c_float(temperature),
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set target acceptance ratio
_MC_Set_Acceptance_Ratio             = _spirit.Parameters_MC_Set_Acceptance_Ratio
_MC_Set_Acceptance_Ratio.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_MC_Set_Acceptance_Ratio.restype     = None
def set_acceptance_ratio(p_state, ratio, idx_image=-1, idx_chain=-1):
    _MC_Set_Acceptance_Ratio(p_state, ctypes.c_float(ratio), idx_image, idx_chain)

## ---------------------------------- Get ----------------------------------

### Get number of iterations and step size
_MC_Get_N_Iterations             = _spirit.Parameters_MC_Get_N_Iterations
_MC_Get_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_MC_Get_N_Iterations.restype     = None
def get_iterations(p_state, idx_image=-1, idx_chain=-1):
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _MC_Get_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

### Get temperature
_MC_Get_Temperature             = _spirit.Parameters_MC_Get_Temperature
_MC_Get_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_MC_Get_Temperature.restype     = ctypes.c_float
def get_temperature(p_state, idx_image=-1, idx_chain=-1):
    return float(_MC_Get_Temperature(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                             ctypes.c_int(idx_chain)))

### Get target acceptance ratio
_MC_Get_Acceptance_Ratio             = _spirit.Parameters_MC_Get_Acceptance_Ratio
_MC_Get_Acceptance_Ratio.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_MC_Get_Acceptance_Ratio.restype     = ctypes.c_float
def get_acceptance_ratio(p_state, idx_image=-1, idx_chain=-1):
    return float(_MC_Get_Acceptance_Ratio(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                    ctypes.c_int(idx_chain)))