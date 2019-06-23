"""
Minimum mode following (MMF)
-------------------------------------------------------------
"""

import spirit.spiritlib as spiritlib
from spirit.io import FILEFORMAT_OVF_TEXT
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

## ---------------------------------- Set ----------------------------------

_MMF_Set_Output_Tag          = _spirit.Parameters_MMF_Set_Output_Tag
_MMF_Set_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_MMF_Set_Output_Tag.restype  = None
def set_output_tag(p_state, tag, idx_image=-1, idx_chain=-1):
    """Set the tag placed in front of output file names.

    If the tag is "<time>", it will be the date-time of the creation of the state."""
    _MMF_Set_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MMF_Set_Output_Folder          = _spirit.Parameters_MMF_Set_Output_Folder
_MMF_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_MMF_Set_Output_Folder.restype  = None
def set_output_folder(p_state, folder, idx_image=-1, idx_chain=-1):
    """Set the folder, where output files are placed."""
    _MMF_Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MMF_Set_Output_General          = _spirit.Parameters_MMF_Set_Output_General
_MMF_Set_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MMF_Set_Output_General.restype  = None
def set_output_general(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    """Set whether to write any output files at all."""
    _MMF_Set_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MMF_Set_Output_Energy          = _spirit.Parameters_MMF_Set_Output_Energy
_MMF_Set_Output_Energy.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MMF_Set_Output_Energy.restype  = None
def set_output_energy(p_state, step=False, archive=True, spin_resolved=False, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    """Set whether to write energy output files.

    - step: whether to write a new file after each set of iterations
    - archive: whether to append to an archive file after each set of iterations
    - spin_resolved: whether to write a file containing the energy of each spin
    - divide_by_nos: whether to divide energies by the number of spins
    - add_readability_lines: whether to separate columns by lines
    """
    _MMF_Set_Output_Energy(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_bool(spin_resolved), ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MMF_Set_Output_Configuration          = _spirit.Parameters_MMF_Set_Output_Configuration
_MMF_Set_Output_Configuration.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
_MMF_Set_Output_Configuration.restype  = None
def set_output_configuration(p_state, step=False, archive=True, filetype=FILEFORMAT_OVF_TEXT, idx_image=-1, idx_chain=-1):
    """Set whether to write spin configuration output files.

    - step: whether to write a new file after each set of iterations
    - archive: whether to append to an archive file after each set of iterations
    - filetype: the format in which the data is written
    """
    _MMF_Set_Output_Configuration(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_int(filetype), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MMF_Set_N_Iterations             = _spirit.Parameters_MMF_Set_N_Iterations
_MMF_Set_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int]
_MMF_Set_N_Iterations.restype     = None
def set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    """Set the number of iterations and how often to log and write output.

    - n_iterations: the maximum number of iterations
    - n_iterations_log: the number of iterations after which status is logged and output written
    """
    _MMF_Set_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                          ctypes.c_int(idx_chain))

_MMF_Set_N_Modes          = _spirit.Parameters_MMF_Set_N_Modes
_MMF_Set_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
_MMF_Set_N_Modes.restype  = None
def set_n_modes(p_state, n_modes, idx_image=-1, idx_chain=-1):
    """Set the number of modes to be calculated at each iteration."""
    _MMF_Set_N_Modes(ctypes.c_void_p(p_state), ctypes.c_int(n_modes),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MMF_Set_N_Mode_Follow          = _spirit.Parameters_MMF_Set_N_Mode_Follow
_MMF_Set_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
_MMF_Set_N_Mode_Follow.restype  = None
def set_n_mode_follow(p_state, n_mode, idx_image=-1, idx_chain=-1):
    """Set the index of the mode to follow."""
    _MMF_Set_N_Mode_Follow(ctypes.c_void_p(p_state), ctypes.c_int(n_mode),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

## ---------------------------------- Get ----------------------------------

_MMF_Get_N_Iterations             = _spirit.Parameters_MMF_Get_N_Iterations
_MMF_Get_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_MMF_Get_N_Iterations.restype     = None
def get_iterations(p_state, idx_image=-1, idx_chain=-1):
    """Returns the maximum number of iterations and the step size."""
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _MMF_Get_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

_MMF_Get_N_Modes          = _spirit.Parameters_MMF_Get_N_Modes
_MMF_Get_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_MMF_Get_N_Modes.restype  = ctypes.c_int
def get_n_modes(p_state, idx_image=-1, idx_chain=-1):
    """Returns the number of modes calculated at each iteration."""
    return int(_MMF_Get_N_Modes(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_MMF_Get_N_Mode_Follow          = _spirit.Parameters_MMF_Get_N_Mode_Follow
_MMF_Get_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_MMF_Get_N_Mode_Follow.restype  = ctypes.c_int
def get_n_mode_follow(p_state, idx_image=-1, idx_chain=-1):
    """Returns the index of the mode which to follow."""
    return int(_MMF_Get_N_Mode_Follow(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))