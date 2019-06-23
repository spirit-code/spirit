"""
Geodesic nudged elastic band (GNEB)
-------------------------------------------------------------
"""

import spirit.spiritlib as spiritlib
from spirit.io import FILEFORMAT_OVF_TEXT
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

# GNEB image types
IMAGE_NORMAL     = 0
"""Regular GNEB image type."""

IMAGE_CLIMBING   = 1
"""Climbing GNEB image type.
Climbing images move towards maxima along the path.
"""

IMAGE_FALLING    = 2
"""Falling GNEB image type.
Falling images move towards the closest minima.
"""

IMAGE_STATIONARY = 3
"""Stationary GNEB image type.
Stationary images are not influenced during a GNEB calculation.
"""

### ---------------------------------- Set ----------------------------------

_GNEB_Set_Output_Tag          = _spirit.Parameters_GNEB_Set_Output_Tag
_GNEB_Set_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Tag.restype  = None
def set_output_tag(p_state, tag, idx_image=-1, idx_chain=-1):
    """Set the tag placed in front of output file names.

    If the tag is "<time>", it will be the date-time of the creation of the state."""
    _GNEB_Set_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Output_Folder          = _spirit.Parameters_GNEB_Set_Output_Folder
_GNEB_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Folder.restype  = None
def set_output_folder(p_state, folder, idx_image=-1, idx_chain=-1):
    """Set the folder, where output files are placed."""
    _GNEB_Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Output_General          = _spirit.Parameters_GNEB_Set_Output_General
_GNEB_Set_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_General.restype  = None
def set_output_general(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    """Set whether to write any output files at all."""
    _GNEB_Set_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Output_Energies          = _spirit.Parameters_GNEB_Set_Output_Energies
_GNEB_Set_Output_Energies.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Energies.restype  = None
def set_output_energies(p_state, step=True, interpolated=True, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    """Set whether to write energy output files.

    - step: whether to write a new file after each set of iterations
    - interpolated: whether to write a file containing interpolated reaction coordinate and energy values
    - divide_by_nos: whether to divide energies by the number of spins
    - add_readability_lines: whether to separate columns by lines
    """
    _GNEB_Set_Output_Energies(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(interpolated),
                        ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Output_Chain          = _spirit.Parameters_GNEB_Set_Output_Chain
_GNEB_Set_Output_Chain.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Output_Chain.restype  = None
def set_output_chain(p_state, step=False, filetype=FILEFORMAT_OVF_TEXT, idx_image=-1, idx_chain=-1):
    """Set whether to write chain output files.

    - step: whether to write a new file after each set of iterations
    - filetype: the format in which the data is written
    """
    _GNEB_Set_Output_Chain(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_int(filetype),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_N_Iterations             = _spirit.Parameters_GNEB_Set_N_Iterations
_GNEB_Set_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int]
_GNEB_Set_N_Iterations.restype     = None
def set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    """Set the number of iterations and how often to log and write output.

    - n_iterations: the maximum number of iterations
    - n_iterations_log: the number of iterations after which status is logged and output written
    """
    _GNEB_Set_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                           ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                           ctypes.c_int(idx_chain))

_GNEB_Set_Convergence           = _spirit.Parameters_GNEB_Set_Convergence
_GNEB_Set_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Convergence.restype   = None
def set_convergence(p_state, convergence, idx_image=-1, idx_chain=-1):
    """Set the convergence limit.

    When the maximum absolute component value of the force drops below this value,
    the calculation is considered converged and will stop.
    """
    _GNEB_Set_Convergence(ctypes.c_void_p(p_state), ctypes.c_float(convergence),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Spring_Constant          = _spirit.Parameters_GNEB_Set_Spring_Constant
_GNEB_Set_Spring_Constant.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Spring_Constant.restype  = None
_GNEB_Set_Spring_Force_Ratio          = _spirit.Parameters_GNEB_Set_Spring_Force_Ratio
_GNEB_Set_Spring_Force_Ratio.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Spring_Force_Ratio.restype  = None
def set_spring_force(p_state, spring_constant=1, ratio=0, idx_image=-1, idx_chain=-1):
    """Set the spring force constant and the ratio between energy and reaction coordinate."""
    _GNEB_Set_Spring_Constant(ctypes.c_void_p(p_state), ctypes.c_float(spring_constant),
                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    _GNEB_Set_Spring_Force_Ratio(ctypes.c_void_p(p_state), ctypes.c_float(ratio),
                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Path_Shortening_Constant           = _spirit.Parameters_GNEB_Set_Path_Shortening_Constant
_GNEB_Set_Path_Shortening_Constant.argtypes  = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Path_Shortening_Constant.restype   = None
def set_path_shortening_constant(p_state, shortening_constant, idx_image=-1, idx_chain=-1):
    """Set the path shortening constant."""
    _GNEB_Set_Path_Shortening_Constant(ctypes.c_void_p(p_state), ctypes.c_float(shortening_constant),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Climbing_Falling             = _spirit.Parameters_GNEB_Set_Climbing_Falling
_GNEB_Set_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_GNEB_Set_Climbing_Falling.restype     = None
def set_climbing_falling(p_state, image_type, idx_image=-1, idx_chain=-1):
    """Set the GNEB image type (see the integers defined above)."""
    _GNEB_Set_Climbing_Falling(ctypes.c_void_p(p_state), ctypes.c_int(image_type),
                               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_GNEB_Set_Image_Type_Automatically             = _spirit.Parameters_GNEB_Set_Image_Type_Automatically
_GNEB_Set_Image_Type_Automatically.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_GNEB_Set_Image_Type_Automatically.restype     = None
def set_image_type_automatically(p_state, idx_chain=-1):
    """Automatically set GNEB image types.

    Minima along the path will be set to falling, maxima to climbing and the rest to regular.
    """
    _GNEB_Set_Image_Type_Automatically(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))

### ---------------------------------- Get ----------------------------------

_GNEB_Get_N_Iterations             = _spirit.Parameters_GNEB_Get_N_Iterations
_GNEB_Get_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                      ctypes.POINTER( ctypes.c_int ), ctypes.c_int]
_GNEB_Get_N_Iterations.restype     = None
def get_iterations(p_state, idx_image=-1, idx_chain=-1):
    """Returns the maximum number of iterations and the step size."""
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _GNEB_Get_N_Iterations(ctypes.c_void_p(p_state), ctypes.pointer(n_iterations),
                           ctypes.pointer(n_iterations_log), ctypes.c_int(idx_image),
                           ctypes.c_int(idx_chain) )
    return int(n_iterations.value), int(n_iterations_log.value)

_GNEB_Get_Convergence           = _spirit.Parameters_GNEB_Get_Convergence
_GNEB_Get_Convergence.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Convergence.restype   = ctypes.c_float
def get_convergence(p_state, idx_image=-1, idx_chain=-1):
    """Returns the convergence value."""
    return float( _GNEB_Get_Convergence(ctypes.c_void_p(p_state),
                                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_GNEB_Get_Spring_Constant          = _spirit.Parameters_GNEB_Get_Spring_Constant
_GNEB_Get_Spring_Constant.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Spring_Constant.restype  = ctypes.c_float
_GNEB_Get_Spring_Force_Ratio          = _spirit.Parameters_GNEB_Get_Spring_Force_Ratio
_GNEB_Get_Spring_Force_Ratio.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Spring_Force_Ratio.restype  = ctypes.c_float
def get_spring_force(p_state, idx_image=-1, idx_chain=-1):
    """Returns the spring force constant and Ratio of energy to reaction coordinate."""
    constant = float(_GNEB_Get_Spring_Constant(ctypes.c_void_p(p_state),
                                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))
    ratio    = float(_GNEB_Get_Spring_Force_Ratio(ctypes.c_void_p(p_state),
                                           ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))
    return constant, ratio

_GNEB_Get_Path_Shortening_Constant           = _spirit.Parameters_GNEB_Get_Path_Shortening_Constant
_GNEB_Get_Path_Shortening_Constant.argtypes  = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Path_Shortening_Constant.restype   = ctypes.c_float
def get_path_shortening_constant(p_state, idx_image=-1, idx_chain=-1):
    """Return the path shortening constant."""
    return float( _GNEB_Get_Path_Shortening_Constant(ctypes.c_void_p(p_state),
                                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_GNEB_Get_Climbing_Falling             = _spirit.Parameters_GNEB_Get_Climbing_Falling
_GNEB_Get_Climbing_Falling.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_GNEB_Get_Climbing_Falling.restype     = ctypes.c_int
def get_climbing_falling(p_state, idx_image=-1, idx_chain=-1):
    """Returns the integer of whether an image is regular, climbing, falling, or stationary.

    The integers are defined above."""
    return int(_GNEB_Get_Climbing_Falling(ctypes.c_void_p(p_state),
                                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_GNEB_Get_N_Energy_Interpolations             = _spirit.Parameters_GNEB_Get_N_Energy_Interpolations
_GNEB_Get_N_Energy_Interpolations.argtypes    = [ctypes.c_void_p, ctypes.c_int]
_GNEB_Get_N_Energy_Interpolations.restype     = ctypes.c_int
def get_n_energy_interpolations(p_state, idx_chain=-1):
    """Returns the number of energy values interpolated between images."""
    return int(_GNEB_Get_N_Energy_Interpolations(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))