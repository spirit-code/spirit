import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

## ---------------------------------- Set ----------------------------------

### Set the output file tag, which is placed in front of all output files of MMF simulations
_Set_MMF_Output_Tag          = _spirit.Parameters_Set_MMF_Output_Tag
_Set_MMF_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_Set_MMF_Output_Tag.restype  = None
def setOutputTag(p_state, tag, idx_image=-1, idx_chain=-1):
    _Set_MMF_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output folder for MMF simulations
_Set_MMF_Output_Folder          = _spirit.Parameters_Set_MMF_Output_Folder
_Set_MMF_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_Set_MMF_Output_Folder.restype  = None
def setOutputFolder(p_state, folder, idx_image=-1, idx_chain=-1):
    _Set_MMF_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration
_Set_MMF_Output_General          = _spirit.Parameters_Set_MMF_Output_General
_Set_MMF_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Set_MMF_Output_General.restype  = None
def setOutputGeneral(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    _Set_MMF_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration for energy files
_Set_MMF_Output_Energy          = _spirit.Parameters_Set_MMF_Output_Energy
_Set_MMF_Output_Energy.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Set_MMF_Output_Energy.restype  = None
def setOutputEnergy(p_state, step=False, archive=True, spin_resolved=False, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    _Set_MMF_Output_Energy(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_bool(spin_resolved), ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set the output configuration for energy files
_Set_MMF_Output_Configuration          = _spirit.Parameters_Set_MMF_Output_Configuration
_Set_MMF_Output_Configuration.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_MMF_Output_Configuration.restype  = None
def setOutputConfiguration(p_state, step=False, archive=True, filetype=6, idx_image=-1, idx_chain=-1):
    _Set_MMF_Output_Configuration(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_int(filetype), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set number of iterations and step size
_Set_MMF_N_Iterations             = _spirit.Parameters_Set_MMF_N_Iterations
_Set_MMF_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int]
_Set_MMF_N_Iterations.restype     = None
def setIterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    _Set_MMF_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                          ctypes.c_int(idx_chain))

### Set number of modes
_Set_MMF_N_Modes          = _spirit.Parameters_Set_MMF_N_Modes
_Set_MMF_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
_Set_MMF_N_Modes.restype  = None
def setNModes(p_state, n_modes, idx_image=-1, idx_chain=-1):
    _Set_MMF_N_Modes(ctypes.c_void_p(p_state), ctypes.c_int(n_modes),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set index of mode to follow
_Set_MMF_N_Mode_Follow          = _spirit.Parameters_Set_MMF_N_Mode_Follow
_Set_MMF_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
_Set_MMF_N_Mode_Follow.restype  = None
def setNModeFollow(p_state, n_mode, idx_image=-1, idx_chain=-1):
    _Set_MMF_N_Mode_Follow(ctypes.c_void_p(p_state), ctypes.c_int(n_mode),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

## ---------------------------------- Get ----------------------------------

### Get number of iterations and step size
_Get_MMF_N_Iterations             = _spirit.Parameters_Get_MMF_N_Iterations
_Get_MMF_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_Get_MMF_N_Iterations.restype     = None
def getIterations(p_state, idx_image=-1, idx_chain=-1):
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _Get_MMF_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

### Get number of modes
_Get_MMF_N_Modes          = _spirit.Parameters_Get_MMF_N_Modes
_Get_MMF_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_MMF_N_Modes.restype  = ctypes.c_int
def getNModes(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_MMF_N_Modes(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get index of mode to follow
_Get_MMF_N_Mode_Follow          = _spirit.Parameters_Get_MMF_N_Mode_Follow
_Get_MMF_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_MMF_N_Mode_Follow.restype  = ctypes.c_int
def getNModeFollow(p_state, idx_image=-1, idx_chain=-1):
    return int(_Get_MMF_N_Mode_Follow(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))