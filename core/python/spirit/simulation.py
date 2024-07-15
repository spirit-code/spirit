"""
Simulation
====================

This module of Spirit is used to run and monitor iterative calculation methods.

If many iterations are called individually, one should use the single shot simulation functionality.
It avoids the allocations etc. involved when a simulation is started and ended and behaves like a
regular simulation, except that the iterations have to be triggered manually.

Note that the VP and LBFGS Solvers are only meant for direct minimization and not for dynamics.
"""

from spirit import spiritlib
from spirit.scalar import scalar
import ctypes

### Load Library
from spirit.spiritlib import _spirit


###     We use a thread for PlayPause, so that KeyboardInterrupt can be forwarded to the CDLL call
###     We might want to think about using PyDLL and about a signal handler in the core library
###     see here: http://stackoverflow.com/questions/14271697/ctrlc-doesnt-interrupt-call-to-shared-library-using-ctypes-in-python


SOLVER_VP = 0
"""Verlet-like velocity projection method."""

SOLVER_SIB = 1
"""Semi-implicit midpoint method B."""

SOLVER_DEPONDT = 2
"""Depondt's Heun-like method."""

SOLVER_HEUN = 3
"""Heun's method."""

SOLVER_RK4 = 4
"""4th order Runge-Kutta method."""

SOLVER_LBFGS_OSO = 5
"""Limited memory Broyden-Fletcher-Goldfarb-Shannon, using exponential transforms."""

SOLVER_LBFGS_Atlas = 6
"""Limited memory Broyden-Fletcher-Goldfarb-Shannon, using stereographic transforms"""

SOLVER_VP_OSO = 7
"""Verlet-like velocity projection method, using exponential transforms."""


METHOD_MC = 0
"""Monte Carlo.

Standard implementation.
"""

METHOD_LLG = 1
"""Landau-Lifshitz-Gilbert.

Can be either a dynamical simulation or an energy minimisation.
Note: the VP solver can *only* minimise.
"""

METHOD_GNEB = 2
"""Geodesic nudged elastic band.

Runs on the entire chain.

As this is a minimisation method, the dynamical solvers perform worse
than those designed for minimisation.
"""

METHOD_MMF = 3
"""Minimum mode following.

As this is a minimisation method, the dynamical solvers perform worse
than those designed for minimisation.
"""

METHOD_EMA = 4
"""Eigenmode analysis.

Applies eigenmodes to the spins of a system.
Depending on parameters, this can be used to calculate the change of a
spin configuration through such a mode or to get a "dynamical" chain
of images corresponding to the movement of the system under the mode.
"""


class simulation_run_info(ctypes.Structure):
    """Contains basic information about a simulation run."""

    _fields_ = [
        ("total_iterations", ctypes.c_int),
        ("total_walltime", ctypes.c_int),
        ("total_ips", scalar),
        ("max_torque", scalar),
        ("_n_history_iteration", ctypes.c_int),
        ("_history_iteration", ctypes.POINTER(ctypes.c_int)),
        ("_n_history_max_torque", ctypes.c_int),
        ("_history_max_torque", ctypes.POINTER(scalar)),
        ("_n_history_energy", ctypes.c_int),
        ("_history_energy", ctypes.POINTER(scalar)),
    ]

    def history_iteration(self):
        return [self._history_iteration[i] for i in range(self._n_history_iteration)]

    def history_max_torque(self):
        return [self._history_max_torque[i] for i in range(self._n_history_max_torque)]

    def history_energy(self):
        return [self._history_energy[i] for i in range(self._n_history_energy)]

    def __del__(self):
        _spirit.free_run_info(self)


### ----- Start methods
### MC
_MC_Start = _spirit.Simulation_MC_Start
_MC_Start.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(simulation_run_info),
    ctypes.c_int,
    ctypes.c_int,
]
_MC_Start.restype = None
### LLG
_LLG_Start = _spirit.Simulation_LLG_Start
_LLG_Start.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(simulation_run_info),
    ctypes.c_int,
    ctypes.c_int,
]
_LLG_Start.restype = None
### GNEB
_GNEB_Start = _spirit.Simulation_GNEB_Start
_GNEB_Start.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(simulation_run_info),
    ctypes.c_int,
]
_GNEB_Start.restype = None
### MMF
_MMF_Start = _spirit.Simulation_MMF_Start
_MMF_Start.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(simulation_run_info),
    ctypes.c_int,
    ctypes.c_int,
]
_MMF_Start.restype = None
### EMA
_EMA_Start = _spirit.Simulation_EMA_Start
_EMA_Start.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.POINTER(simulation_run_info),
    ctypes.c_int,
    ctypes.c_int,
]
_EMA_Start.restype = None


### ----- Wrapper
def start(
    p_state,
    method_type,
    solver_type=None,
    n_iterations=-1,
    n_iterations_log=-1,
    single_shot=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Start any kind of iterative calculation method.

    - `method_type`: one of the integers defined above
    - `solver_type`: only used for LLG, GNEB and MMF methods (default: None)
    - `n_iterations`: the maximum number of iterations that will be performed (default: take from parameters)
    - `n_iterations_log`: the number of iterations after which to log the status and write output (default: take from parameters)
    - `single_shot`: if set to `True`, iterations have to be triggered individually
    - `idx_image`: the image on which to run the calculation (default: active image). Not used for GNEB

    returns a `simulation_run_info` object.
    """

    info = simulation_run_info()

    if method_type == METHOD_MC:
        spiritlib.wrap_function(
            _MC_Start,
            [
                ctypes.c_void_p(p_state),
                ctypes.c_int(n_iterations),
                ctypes.c_int(n_iterations_log),
                ctypes.c_bool(single_shot),
                ctypes.pointer(info),
                ctypes.c_int(idx_image),
                ctypes.c_int(idx_chain),
            ],
        )
    elif method_type == METHOD_LLG:
        spiritlib.wrap_function(
            _LLG_Start,
            [
                ctypes.c_void_p(p_state),
                ctypes.c_int(solver_type),
                ctypes.c_int(n_iterations),
                ctypes.c_int(n_iterations_log),
                ctypes.c_bool(single_shot),
                ctypes.pointer(info),
                ctypes.c_int(idx_image),
                ctypes.c_int(idx_chain),
            ],
        )
    elif method_type == METHOD_GNEB:
        spiritlib.wrap_function(
            _GNEB_Start,
            [
                ctypes.c_void_p(p_state),
                ctypes.c_int(solver_type),
                ctypes.c_int(n_iterations),
                ctypes.c_int(n_iterations_log),
                ctypes.c_bool(single_shot),
                ctypes.pointer(info),
                ctypes.c_int(idx_chain),
            ],
        )
    elif method_type == METHOD_MMF:
        spiritlib.wrap_function(
            _MMF_Start,
            [
                ctypes.c_void_p(p_state),
                ctypes.c_int(solver_type),
                ctypes.c_int(n_iterations),
                ctypes.c_int(n_iterations_log),
                ctypes.c_bool(single_shot),
                ctypes.pointer(info),
                ctypes.c_int(idx_image),
                ctypes.c_int(idx_chain),
            ],
        )
    elif method_type == METHOD_EMA:
        spiritlib.wrap_function(
            _EMA_Start,
            [
                ctypes.c_void_p(p_state),
                ctypes.c_int(n_iterations),
                ctypes.c_int(n_iterations_log),
                ctypes.c_bool(single_shot),
                ctypes.pointer(info),
                ctypes.c_int(idx_image),
                ctypes.c_int(idx_chain),
            ],
        )
    else:
        print("Invalid method_type passed to simulation.start...")

    return info
    # _Start(ctypes.c_void_p(p_state), ctypes.c_char_p(method_type),
    #            ctypes.c_char_p(solver_type), ctypes.c_int(n_iterations),
    #            ctypes.c_int(n_iterations_log), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


_SingleShot = _spirit.Simulation_SingleShot
_SingleShot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_SingleShot.restype = None


def single_shot(p_state, idx_image=-1, idx_chain=-1):
    """Perform a single iteration.

    In order to use this, a single shot simulation must be running on the corresponding image or chain.
    """
    spiritlib.wrap_function(
        _SingleShot,
        [ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)],
    )


_N_Shot = _spirit.Simulation_N_Shot
_N_Shot.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_N_Shot.restype = None


def n_shot(p_state, N, idx_image=-1, idx_chain=-1):
    """Perform a single iteration.

    In order to use this, a single shot simulation must be running on the corresponding image or chain.
    """
    spiritlib.wrap_function(
        _N_Shot,
        [
            ctypes.c_void_p(p_state),
            ctypes.c_int(N),
            ctypes.c_int(idx_image),
            ctypes.c_int(idx_chain),
        ],
    )


_Stop = _spirit.Simulation_Stop
_Stop.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Stop.restype = None


def stop(p_state, idx_image=-1, idx_chain=-1):
    """Stop the simulation running on an image or chain."""
    _Stop(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


_Stop_All = _spirit.Simulation_Stop_All
_Stop_All.argtypes = [ctypes.c_void_p]
_Stop_All.restype = None


def stop_all(p_state):
    """Stop all simulations running anywhere."""
    _Stop_All(ctypes.c_void_p(p_state))


_Running_On_Image = _spirit.Simulation_Running_On_Image
_Running_On_Image.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Running_On_Image.restype = ctypes.c_bool


def running_on_image(p_state, idx_image=-1, idx_chain=-1):
    """Check if a simulation is running on a specific image."""
    return bool(
        _Running_On_Image(
            ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)
        )
    )


_Running_On_Chain = _spirit.Simulation_Running_On_Chain
_Running_On_Chain.argtypes = [ctypes.c_void_p, ctypes.c_int]
_Running_On_Chain.restype = ctypes.c_bool


def running_on_chain(p_state, idx_chain=-1):
    """Check if a simulation is running across a specific chain."""
    return bool(_Running_On_Chain(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain)))


_Running_Anywhere_On_Chain = _spirit.Simulation_Running_Anywhere_On_Chain
_Running_Anywhere_On_Chain.argtypes = [ctypes.c_void_p, ctypes.c_int]
_Running_Anywhere_On_Chain.restype = ctypes.c_bool


def running_anywhere_on_chain(p_state, idx_chain=-1):
    """Check if any simulation running on any image of - or the entire - chain."""
    return bool(
        _Running_Anywhere_On_Chain(ctypes.c_void_p(p_state), ctypes.c_int(idx_chain))
    )


_Get_IterationsPerSecond = _spirit.Simulation_Get_IterationsPerSecond
_Get_IterationsPerSecond.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_IterationsPerSecond.restype = scalar


def get_iterations_per_second(p_state, idx_image=-1, idx_chain=-1):
    """Returns the current estimation of the number of iterations per second."""
    return float(
        _Get_IterationsPerSecond(
            ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)
        )
    )


_Get_MaxTorqueNorm = _spirit.Simulation_Get_MaxTorqueNorm
_Get_MaxTorqueNorm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_MaxTorqueNorm.restype = scalar


def get_max_torque_norm(p_state, idx_image=-1, idx_chain=-1):
    """Returns the current maximum norm of the torque acting on any spin."""
    return float(
        _Get_MaxTorqueNorm(
            ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)
        )
    )


_Get_Time = _spirit.Simulation_Get_Time
_Get_Time.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Time.restype = scalar


def get_time(p_state, idx_image=-1, idx_chain=-1):
    """If an LLG simulation is running returns the cumulatively summed time steps `dt`, otherwise returns 0"""
    return _Get_Time(
        ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)
    )


_Get_Wall_Time = _spirit.Simulation_Get_Wall_Time
_Get_Wall_Time.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Wall_Time.restype = ctypes.c_int


def get_wall_time(p_state, idx_image=-1, idx_chain=-1):
    """Returns the current maximum norm of the torque acting on any spin."""
    return int(
        _Get_Wall_Time(
            ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)
        )
    )
