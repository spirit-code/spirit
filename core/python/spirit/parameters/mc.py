"""
Monte Carlo (MC)
-------------------------------------------------------------
"""

import spirit.spiritlib as spiritlib
from spirit.io import FILEFORMAT_OVF_TEXT
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

## ---------------------------------- Set ----------------------------------

_MC_Set_Output_Tag          = _spirit.Parameters_MC_Set_Output_Tag
_MC_Set_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Tag.restype  = None
def set_output_tag(p_state, tag, idx_image=-1, idx_chain=-1):
    """Set the tag placed in front of output file names.

    If the tag is "<time>", it will be the date-time of the creation of the state."""
    _MC_Set_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MC_Set_Output_Folder          = _spirit.Parameters_MC_Set_Output_Folder
_MC_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Folder.restype  = None
def set_output_folder(p_state, folder, idx_image=-1, idx_chain=-1):
    """Set the folder, where output files are placed."""
    _MC_Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MC_Set_Output_General          = _spirit.Parameters_MC_Set_Output_General
_MC_Set_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MC_Set_Output_General.restype  = None
def set_output_general(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    """Set whether to write any output files at all."""
    _MC_Set_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MC_Set_Output_Energy          = _spirit.Parameters_MC_Set_Output_Energy
_MC_Set_Output_Energy.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Energy.restype  = None
def set_output_energy(p_state, step=False, archive=True, spin_resolved=False, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    """Set whether to write energy output files.

    - step: whether to write a new file after each set of iterations
    - archive: whether to append to an archive file after each set of iterations
    - spin_resolved: whether to write a file containing the energy of each spin
    - divide_by_nos: whether to divide energies by the number of spins
    - add_readability_lines: whether to separate columns by lines
    """
    _MC_Set_Output_Energy(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_bool(spin_resolved), ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MC_Set_Output_Configuration          = _spirit.Parameters_MC_Set_Output_Configuration
_MC_Set_Output_Configuration.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
_MC_Set_Output_Configuration.restype  = None
def set_output_configuration(p_state, step=False, archive=True, filetype=FILEFORMAT_OVF_TEXT, idx_image=-1, idx_chain=-1):
    """Set whether to write spin configuration output files.

    - step: whether to write a new file after each set of iterations
    - archive: whether to append to an archive file after each set of iterations
    - filetype: the format in which the data is written
    """
    _MC_Set_Output_Configuration(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_int(filetype), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MC_Set_N_Iterations             = _spirit.Parameters_MC_Set_N_Iterations
_MC_Set_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int]
_MC_Set_N_Iterations.restype     = None
def set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    """Set the number of iterations and how often to log and write output.

    - n_iterations: the maximum number of iterations
    - n_iterations_log: the number of iterations after which status is logged and output written
    """
    _MC_Set_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                          ctypes.c_int(idx_chain))

_MC_Set_Temperature             = _spirit.Parameters_MC_Set_Temperature
_MC_Set_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_MC_Set_Temperature.restype     = None
def set_temperature(p_state, temperature, idx_image=-1, idx_chain=-1):
    """Set the global base temperature [K]."""
    _MC_Set_Temperature(ctypes.c_void_p(p_state), ctypes.c_float(temperature),
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_MC_Set_Metropolis_Cone             = _spirit.Parameters_MC_Set_Metropolis_Cone
_MC_Set_Metropolis_Cone.argtypes    = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float,
                                            ctypes.c_bool, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_MC_Set_Metropolis_Cone.restype     = None
def set_metropolis_cone(p_state, use_cone=True, cone_angle=40, use_adaptive_cone=True, target_acceptance_ratio=0.5, idx_image=-1, idx_chain=-1):
    """Configure the Metropolis parameters.

    - use_cone: whether to displace the spins within a cone (otherwise: on the entire unit sphere)
    - cone_angle: the opening angle within which the spin is placed
    - use_adaptive_cone: automatically adapt the cone angle to achieve the set acceptance ratio
    - target_acceptance_ratio: target acceptance ratio for the adaptive cone algorithm
    """
    _MC_Set_Metropolis_Cone(p_state, ctypes.c_bool(use_cone), ctypes.c_float(cone_angle), ctypes.c_bool(use_adaptive_cone), ctypes.c_float(target_acceptance_ratio), idx_image, idx_chain)

## ---------------------------------- Get ----------------------------------

_MC_Get_N_Iterations             = _spirit.Parameters_MC_Get_N_Iterations
_MC_Get_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_MC_Get_N_Iterations.restype     = None
def get_iterations(p_state, idx_image=-1, idx_chain=-1):
    """Returns the maximum number of iterations and the step size."""
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _MC_Get_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

_MC_Get_Temperature             = _spirit.Parameters_MC_Get_Temperature
_MC_Get_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_MC_Get_Temperature.restype     = ctypes.c_float
def get_temperature(p_state, idx_image=-1, idx_chain=-1):
    """Returns the global base temperature [K]."""
    return float(_MC_Get_Temperature(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                             ctypes.c_int(idx_chain)))

_MC_Get_Metropolis_Cone             = _spirit.Parameters_MC_Get_Metropolis_Cone
_MC_Get_Metropolis_Cone.argtypes    = [ ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_float),
                                        ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int, ctypes.c_int ]
_MC_Get_Metropolis_Cone.restype     = None
def get_metropolis_cone(p_state, idx_image=-1, idx_chain=-1):
    """Returns the Metropolis algorithm configuration.

    - whether the spins are displaced within a cone (otherwise: on the entire unit sphere)
    - the opening angle within which the spin is placed
    - whether the cone angle is automatically adapted to achieve the set acceptance ratio
    - target acceptance ratio for the adaptive cone algorithm
    """
    use_cone = ctypes.c_bool()
    cone_angle = ctypes.c_float()
    use_adaptive_cone = ctypes.c_bool()
    target_acceptance_ratio = ctypes.c_float()
    _MC_Get_Metropolis_Cone(ctypes.c_void_p(p_state),
                ctypes.pointer(use_cone), ctypes.pointer(cone_angle),
                ctypes.pointer(use_adaptive_cone), ctypes.pointer(target_acceptance_ratio),
                ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return bool(use_cone), float(cone_angle), bool(use_adaptive_cone), float(target_acceptance_ratio)