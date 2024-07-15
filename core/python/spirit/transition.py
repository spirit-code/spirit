"""
Transition
====================
"""

from spirit.scalar import scalar
import ctypes

### Load Library
from spirit.spiritlib import _spirit

_Homogeneous = _spirit.Transition_Homogeneous
_Homogeneous.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Homogeneous.restype = None


def homogeneous(p_state, idx_1, idx_2, idx_chain=-1):
    """Generate homogeneous transition between two images of a chain."""
    _Homogeneous(
        ctypes.c_void_p(p_state),
        ctypes.c_int(idx_1),
        ctypes.c_int(idx_2),
        ctypes.c_int(idx_chain),
    )


_Homogeneous_Interpolate = _spirit.Transition_Homogeneous_Insert_Interpolated
_Homogeneous_Interpolate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Homogeneous_Interpolate.restype = None


def homogeneous_insert_interpolated(p_state, n_interpolate, idx_chain=-1):
    """Make chain denser by inserting n_interpolate images between all images."""
    _Homogeneous_Interpolate(
        ctypes.c_void_p(p_state), ctypes.c_int(n_interpolate), ctypes.c_int(idx_chain)
    )


_Add_Noise_Temperature = _spirit.Transition_Add_Noise_Temperature
_Add_Noise_Temperature.argtypes = [
    ctypes.c_void_p,
    scalar,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_Add_Noise_Temperature.restype = None


def add_noise(p_state, temperature, idx_1, idx_2, idx_chain=-1):
    """Add some temperature-scaled noise to a transition between two images of a chain."""
    _Add_Noise_Temperature(
        ctypes.c_void_p(p_state),
        scalar(temperature),
        ctypes.c_int(idx_1),
        ctypes.c_int(idx_2),
        ctypes.c_int(idx_chain),
    )
