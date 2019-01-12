"""
Landau-Lifshitz-Gilbert (LLG)
-------------------------------------------------------------
"""

import spirit.spiritlib as spiritlib
from spirit.io import FILEFORMAT_OVF_TEXT
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

### ---------------------------------- Set ----------------------------------

_LLG_Set_Output_Tag          = _spirit.Parameters_LLG_Set_Output_Tag
_LLG_Set_Output_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                ctypes.c_int, ctypes.c_int]
_LLG_Set_Output_Tag.restype  = None
def set_output_tag(p_state, tag, idx_image=-1, idx_chain=-1):
    """Set the tag placed in front of output file names.

    If the tag is "<time>", it will be the date-time of the creation of the state."""
    _LLG_Set_Output_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_Output_Folder          = _spirit.Parameters_LLG_Set_Output_Folder
_LLG_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                    ctypes.c_int, ctypes.c_int]
_LLG_Set_Output_Folder.restype  = None
def set_output_folder(p_state, folder, idx_image=-1, idx_chain=-1):
    """Set the folder, where output files are placed."""
    _LLG_Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(folder.encode('utf-8')),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_Output_General          = _spirit.Parameters_LLG_Set_Output_General
_LLG_Set_Output_General.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_LLG_Set_Output_General.restype  = None
def set_output_general(p_state, any=True, initial=False, final=False, idx_image=-1, idx_chain=-1):
    """Set whether to write any output files at all."""
    _LLG_Set_Output_General(ctypes.c_void_p(p_state), ctypes.c_bool(any), ctypes.c_bool(initial),
                        ctypes.c_bool(final), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_Output_Energy          = _spirit.Parameters_LLG_Set_Output_Energy
_LLG_Set_Output_Energy.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_bool,
                                    ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_LLG_Set_Output_Energy.restype  = None
def set_output_energy(p_state, step=False, archive=True, spin_resolved=False, divide_by_nos=True, add_readability_lines=True, idx_image=-1, idx_chain=-1):
    """Set whether to write energy output files.

    - step: whether to write a new file after each set of iterations
    - archive: whether to append to an archive file after each set of iterations
    - spin_resolved: whether to write a file containing the energy of each spin
    - divide_by_nos: whether to divide energies by the number of spins
    - add_readability_lines: whether to separate columns by lines
    """
    _LLG_Set_Output_Energy(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_bool(spin_resolved), ctypes.c_bool(divide_by_nos), ctypes.c_bool(add_readability_lines),
                        ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_Output_Configuration          = _spirit.Parameters_LLG_Set_Output_Configuration
_LLG_Set_Output_Configuration.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_bool,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LLG_Set_Output_Configuration.restype  = None
def set_output_configuration(p_state, step=False, archive=True, filetype=FILEFORMAT_OVF_TEXT, idx_image=-1, idx_chain=-1):
    """Set whether to write spin configuration output files.

    - step: whether to write a new file after each set of iterations
    - archive: whether to append to an archive file after each set of iterations
    - filetype: the format in which the data is written
    """
    _LLG_Set_Output_Configuration(ctypes.c_void_p(p_state), ctypes.c_bool(step), ctypes.c_bool(archive),
                        ctypes.c_int(filetype), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_N_Iterations             = _spirit.Parameters_LLG_Set_N_Iterations
_LLG_Set_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int]
_LLG_Set_N_Iterations.restype     = None
def set_iterations(p_state, n_iterations, n_iterations_log, idx_image=-1, idx_chain=-1):
    """Set the number of iterations and how often to log and write output.

    - n_iterations: the maximum number of iterations
    - n_iterations_log: the number of iterations after which status is logged and output written
    """
    _LLG_Set_N_Iterations(ctypes.c_void_p(p_state), ctypes.c_int(n_iterations),
                          ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image),
                          ctypes.c_int(idx_chain))

_LLG_Set_Direct_Minimization            = _spirit.Parameters_LLG_Set_Direct_Minimization
_LLG_Set_Direct_Minimization.argtypes   = [ctypes.c_void_p, ctypes.c_bool,
                                           ctypes.c_int, ctypes.c_int ]
_LLG_Set_Direct_Minimization.restype    = None
def set_direct_minimization(p_state, use_minimization, idx_image=-1, idx_chain=-1):
    """Set whether to minimise the energy without precession.

    This only influences dynamics solvers, which will then perform pseudodynamics,
    simulating only the damping part of the LLG equation.
    """
    _LLG_Set_Direct_Minimization(ctypes.c_void_p(p_state), ctypes.c_bool(use_minimization),
                                 ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_Convergence            = _spirit.Parameters_LLG_Set_Convergence
_LLG_Set_Convergence.argtypes   = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int ]
_LLG_Set_Convergence.restype    = None
def set_convergence(p_state, convergence, idx_image=-1, idx_chain=-1):
    """Set the convergence limit.

    When the maximum absolute component value of the force drops below this value,
    the calculation is considered converged and will stop.
    """
    _LLG_Set_Convergence(ctypes.c_void_p(p_state), ctypes.c_float(convergence),
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_Time_Step             = _spirit.Parameters_LLG_Set_Time_Step
_LLG_Set_Time_Step.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_LLG_Set_Time_Step.restype     = None
def set_timestep(p_state, dt, idx_image=-1, idx_chain=-1):
    """Set the time step [ps] for the calculation."""
    _LLG_Set_Time_Step(p_state, ctypes.c_float(dt), idx_image, idx_chain)

_LLG_Set_Damping             = _spirit.Parameters_LLG_Set_Damping
_LLG_Set_Damping.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_LLG_Set_Damping.restype     = None
def set_damping(p_state, damping, idx_image=-1, idx_chain=-1):
    """Set the Gilbert damping parameter [unitless]."""
    _LLG_Set_Damping(ctypes.c_void_p(p_state), ctypes.c_float(damping),
                     ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_LLG_Set_STT             = _spirit.Parameters_LLG_Set_STT
_LLG_Set_STT.argtypes    = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float,
                            ctypes.POINTER(3*ctypes.c_float), ctypes.c_int, ctypes.c_int]
_LLG_Set_STT.restype     = None
def set_stt(p_state, use_gradient, magnitude, direction, idx_image=-1, idx_chain=-1):
    """Set the spin current configuration.

    - use_gradient: `True`: use the spatial gradient, `False`: monolayer approximation
    - magnitude: current strength
    - direction: current direction or polarisation direction, array of shape (3)
    """
    vec3 = ctypes.c_float * 3
    direction = vec3(*direction)
    ctypes.cast( direction, ctypes.POINTER(vec3))
    _LLG_Set_STT(ctypes.c_void_p(p_state), ctypes.c_bool(use_gradient),
                 ctypes.c_float(magnitude), direction, ctypes.c_int(idx_image),
                 ctypes.c_int(idx_chain))

_LLG_Set_Temperature             = _spirit.Parameters_LLG_Set_Temperature
_LLG_Set_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_LLG_Set_Temperature.restype     = None
_LLG_Set_Temperature_Gradient             = _spirit.Parameters_LLG_Set_Temperature_Gradient
_LLG_Set_Temperature_Gradient.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(3*ctypes.c_float),
                                             ctypes.c_int, ctypes.c_int]
_LLG_Set_Temperature_Gradient.restype     = None
def set_temperature(p_state, temperature, gradient_inclination=0, gradient_direction=[1.0,0.0,0.0], idx_image=-1, idx_chain=-1):
    """Set the temperature configuration.

    - temperature: homogeneous base temperature [K]
    - gradient_inclination: inclination of the temperature gradient [K/a]
    - gradient_direction: direction of the temperature gradient, array of shape (3)
    """
    _LLG_Set_Temperature(ctypes.c_void_p(p_state), ctypes.c_float(temperature),
                         ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    vec3 = ctypes.c_float * 3
    gradient_direction = vec3(*gradient_direction)
    ctypes.cast( gradient_direction, ctypes.POINTER(vec3))
    _LLG_Set_Temperature_Gradient(ctypes.c_void_p(p_state), ctypes.c_float(gradient_inclination), gradient_direction,
                                  ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### ---------------------------------- Get ----------------------------------

_LLG_Get_N_Iterations             = _spirit.Parameters_LLG_Get_N_Iterations
_LLG_Get_N_Iterations.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_int ),
                                     ctypes.POINTER( ctypes.c_int ), ctypes.c_int, ctypes.c_int]
_LLG_Get_N_Iterations.restype     = None
def get_iterations(p_state, idx_image=-1, idx_chain=-1):
    """Returns the maximum number of iterations and the step size."""
    n_iterations = ctypes.c_int()
    n_iterations_log = ctypes.c_int()
    _LLG_Get_N_Iterations(p_state, ctypes.pointer(n_iterations), ctypes.pointer(n_iterations_log),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_iterations.value), int(n_iterations_log.value)

_LLG_Get_Direct_Minimization            = _spirit.Parameters_LLG_Get_Direct_Minimization
_LLG_Get_Direct_Minimization.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int ]
_LLG_Get_Direct_Minimization.restype    = ctypes.c_bool
def get_direct_minimization(p_state, idx_image=-1, idx_chain=-1):
    """Returns whether only energy minimisation will be performed."""
    return float(_LLG_Get_Direct_Minimization(ctypes.c_void_p(p_state),
                                              ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_LLG_Get_Convergence            = _spirit.Parameters_LLG_Get_Convergence
_LLG_Get_Convergence.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int ]
_LLG_Get_Convergence.restype    = ctypes.c_float
def get_convergence(p_state, idx_image=-1, idx_chain=-1):
    """Returns the convergence value."""
    return float(_LLG_Get_Convergence(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                      ctypes.c_int(idx_chain)))

_LLG_Get_Time_Step             = _spirit.Parameters_LLG_Get_Time_Step
_LLG_Get_Time_Step.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LLG_Get_Time_Step.restype     = ctypes.c_float
def get_timestep(p_state, idx_image=-1, idx_chain=-1):
    """Returns the time step [ps]."""
    return float(_LLG_Get_Time_Step(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                    ctypes.c_int(idx_chain)))

_LLG_Get_Damping             = _spirit.Parameters_LLG_Get_Damping
_LLG_Get_Damping.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LLG_Get_Damping.restype     = ctypes.c_float
def get_damping(p_state, idx_image=-1, idx_chain=-1):
    """Returns the Gilbert damping parameter."""
    return float(_LLG_Get_Damping(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                            ctypes.c_int(idx_chain)))

_LLG_Get_STT             = _spirit.Parameters_LLG_Get_STT
_LLG_Get_STT.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_bool ),
                            ctypes.POINTER( ctypes.c_float ), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_int, ctypes.c_int]
_LLG_Get_STT.restype     = None
def get_stt(p_state, idx_image=-1, idx_chain=-1):
    """Returns the spin current configuration.

    - magnitude
    - direction, array of shape (3)
    - whether the spatial gradient is used
    """
    direction = (3*ctypes.c_float)()
    use_gradient = ctypes.c_bool()
    magnitude = ctypes.c_float()
    _LLG_Get_STT(ctypes.c_void_p(p_state), ctypes.pointer(use_gradient),
                 ctypes.pointer(magnitude), direction, ctypes.c_int(idx_image),
                 ctypes.c_int(idx_chain))
    return float(magnitude.value), [float(i) for i in direction], bool(use_gradient)

_LLG_Get_Temperature             = _spirit.Parameters_LLG_Get_Temperature
_LLG_Get_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LLG_Get_Temperature.restype     = ctypes.c_float
_LLG_Get_Temperature_Gradient             = _spirit.Parameters_LLG_Get_Temperature_Gradient
_LLG_Get_Temperature_Gradient.argtypes    = [ctypes.c_void_p, ctypes.POINTER( ctypes.c_float ),
                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_LLG_Get_Temperature_Gradient.restype     = None
def get_temperature(p_state, idx_image=-1, idx_chain=-1):
    """Returns the temperature configuration.

    - global base temperature [K]
    - inclination of the temperature gradient [K/a]
    - direction of the temperature gradient, array of shape (3)
    """
    temperature = float(_LLG_Get_Temperature(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                                             ctypes.c_int(idx_chain)))
    gradient_inclination = ctypes.c_float()
    gradient_direction = (3*ctypes.c_float)()
    _LLG_Get_Temperature_Gradient(ctypes.c_void_p(p_state), ctypes.pointer(gradient_inclination),
                                  gradient_direction, ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return temperature, gradient_inclination, gradient_direction
