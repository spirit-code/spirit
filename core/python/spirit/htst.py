"""
HTST
====================

Harmonic transition state theory.

Note that `calculate_prefactor` needs to be called before using any of the getter functions.
"""

from spirit import system
from spirit.scalar import scalar
import ctypes
import numpy as np

### Load Library
from spirit.spiritlib import _spirit


_Calculate = _spirit.HTST_Calculate
_Calculate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_int,
]
_Calculate.restype = scalar


def calculate(
    p_state,
    idx_image_minimum,
    idx_image_sp,
    n_eigenmodes_keep=-1,
    sparse=False,
    idx_chain=-1,
):
    """Performs an HTST calculation and returns rate prefactor.

    *Note:* this function must be called before any of the getters.
    """
    return _Calculate(
        p_state, idx_image_minimum, idx_image_sp, n_eigenmodes_keep, sparse, idx_chain
    )


### Get HTST transition rate components
_Get_Info = _spirit.HTST_Get_Info
_Get_Info.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
_Get_Info.restype = None


def get_info(p_state, idx_chain=-1):
    """Returns a set of HTST information:

    - the exponent of the temperature-dependence
    - me
    - Omega_0
    - s
    - zero mode volume at the minimum
    - zero mode volume at the saddle point
    - dynamical prefactor
    - full rate prefactor (without temperature dependent part)
    """
    temperature_exponent = scalar()
    me = scalar()
    Omega_0 = scalar()
    s = scalar()
    volume_min = scalar()
    volume_sp = scalar()
    prefactor_dynamical = scalar()
    prefactor = scalar()
    n_eigenmodes_keep = ctypes.c_int()

    _Get_Info(
        ctypes.c_void_p(p_state),
        ctypes.pointer(temperature_exponent),
        ctypes.pointer(me),
        ctypes.pointer(Omega_0),
        ctypes.pointer(s),
        ctypes.pointer(volume_min),
        ctypes.pointer(volume_sp),
        ctypes.pointer(prefactor_dynamical),
        ctypes.pointer(prefactor),
        ctypes.pointer(n_eigenmodes_keep),
        ctypes.c_int(idx_chain),
    )

    return (
        temperature_exponent.value,
        me.value,
        Omega_0.value,
        s.value,
        volume_min.value,
        volume_sp.value,
        prefactor_dynamical.value,
        prefactor.value,
    )


def get_info_dict(p_state, idx_chain=-1):
    """Returns a set of HTST information in a dictionary:

    - the exponent of the temperature-dependence
    - me
    - Omega_0
    - s
    - zero mode volume at the minimum
    - zero mode volume at the saddle point
    - dynamical prefactor
    - full rate prefactor (without temperature dependent part)
    """
    temperature_exponent = scalar()
    me = scalar()
    Omega_0 = scalar()
    s = scalar()
    volume_min = scalar()
    volume_sp = scalar()
    prefactor_dynamical = scalar()
    prefactor = scalar()
    n_eigenmodes_keep = ctypes.c_int()

    _Get_Info(
        ctypes.c_void_p(p_state),
        ctypes.pointer(temperature_exponent),
        ctypes.pointer(me),
        ctypes.pointer(Omega_0),
        ctypes.pointer(s),
        ctypes.pointer(volume_min),
        ctypes.pointer(volume_sp),
        ctypes.pointer(prefactor_dynamical),
        ctypes.pointer(prefactor),
        ctypes.pointer(n_eigenmodes_keep),
        ctypes.c_int(idx_chain),
    )

    return {
        "temperature_exponent": temperature_exponent.value,
        "me": me.value,
        "Omega_0": Omega_0.value,
        "s": s.value,
        "volume_min": volume_min.value,
        "volume_sp": volume_sp.value,
        "prefactor_dynamical": prefactor_dynamical.value,
        "prefactor": prefactor.value,
        "n_eigenmodes_keep": n_eigenmodes_keep.value,
    }


_Get_Eigenvalues_Min = _spirit.HTST_Get_Eigenvalues_Min
_Get_Eigenvalues_Min.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.c_int,
]
_Get_Eigenvalues_Min.restype = None


def get_eigenvalues_min(p_state, idx_chain=-1):
    """Returns the eigenvalues at the minimum with `shape(2*nos)`."""
    nos = system.get_nos(p_state, -1, idx_chain)
    eigenvalues_min = (2 * nos * scalar)()
    _Get_Eigenvalues_Min(
        ctypes.c_void_p(p_state), eigenvalues_min, ctypes.c_int(idx_chain)
    )
    return eigenvalues_min


_Get_Eigenvectors_Min = _spirit.HTST_Get_Eigenvectors_Min
_Get_Eigenvectors_Min.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.c_int,
]
_Get_Eigenvectors_Min.restype = None


def get_eigenvectors_min(p_state, idx_chain=-1):
    """Returns a numpy array view to the eigenvectors at the minimum with `shape(n_eigenmodes_keep, 2*nos)`."""
    n_modes = get_info_dict(p_state)["n_eigenmodes_keep"]
    nos = system.get_nos(p_state, -1, idx_chain)
    eigenvectors_min = (2 * nos * n_modes * scalar)()

    ArrayType = scalar * (2 * nos * n_modes)
    ev_list = [] * (2 * nos * n_modes)
    _ev_buffer = ArrayType(*ev_list)

    _Get_Eigenvectors_Min(ctypes.c_void_p(p_state), _ev_buffer, ctypes.c_int(idx_chain))

    ev_array = np.array(_ev_buffer)
    ev_view = ev_array.view()
    ev_view.shape = (n_modes, 2 * nos)

    return ev_view


_Get_Eigenvalues_SP = _spirit.HTST_Get_Eigenvalues_SP
_Get_Eigenvalues_SP.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.c_int,
]
_Get_Eigenvalues_SP.restype = None


def get_eigenvalues_sp(p_state, idx_chain=-1):
    """Returns the eigenvalues at the saddle point with `shape(2*nos)`."""
    nos = system.get_nos(p_state, -1, idx_chain)
    eigenvalues_sp = (2 * nos * scalar)()
    _Get_Eigenvalues_SP(
        ctypes.c_void_p(p_state), eigenvalues_sp, ctypes.c_int(idx_chain)
    )
    return eigenvalues_sp


_Get_Eigenvectors_SP = _spirit.HTST_Get_Eigenvectors_SP
_Get_Eigenvectors_SP.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.c_int,
]
_Get_Eigenvectors_SP.restype = None


def get_eigenvectors_sp(p_state, idx_chain=-1):
    """Returns a numpy array view to the eigenvectors at the saddle point with `shape(n_eigenmodes_keep, 2*nos)`."""

    n_modes = get_info_dict(p_state)["n_eigenmodes_keep"]
    nos = system.get_nos(p_state, -1, idx_chain)

    ArrayType = scalar * (2 * nos * n_modes)
    ev_list = [] * (2 * nos * n_modes)
    _ev_buffer = ArrayType(*ev_list)

    _Get_Eigenvectors_SP(ctypes.c_void_p(p_state), _ev_buffer, ctypes.c_int(idx_chain))

    ev_array = np.array(_ev_buffer)
    ev_view = ev_array.view()
    ev_view.shape = (n_modes, 2 * nos)

    return ev_view


_Get_Velocities = _spirit.HTST_Get_Velocities
_Get_Velocities.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.c_int,
]
_Get_Velocities.restype = None


def get_velocities(p_state, idx_chain=-1):
    """Returns the velocities perpendicular to the dividing surface with `shape(2*nos)`."""
    nos = system.get_nos(p_state, -1, idx_chain)
    velocities = (2 * nos * scalar)()
    _Get_Velocities(ctypes.c_void_p(p_state), velocities, ctypes.c_int(idx_chain))
    return velocities
