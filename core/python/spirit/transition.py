"""
Transition
====================
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

_Homogeneous             = _spirit.Transition_Homogeneous
_Homogeneous.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Homogeneous.restype     = None
def homogeneous(p_state, idx_1, idx_2, idx_chain=-1):
    """Generate homogeneous transition between two images of a chain."""
    _Homogeneous(ctypes.c_void_p(p_state), ctypes.c_int(idx_1), ctypes.c_int(idx_2),
                 ctypes.c_int(idx_chain))

_Add_Noise_Temperature             = _spirit.Transition_Add_Noise_Temperature
_Add_Noise_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int]
_Add_Noise_Temperature.restype     = None
def add_noise(p_state, temperature, idx_1, idx_2, idx_chain=-1):
    """Add some temperature-scaled noise to a transition between two images of a chain."""
    _Add_Noise_Temperature(ctypes.c_void_p(p_state), ctypes.c_float(temperature),
                           ctypes.c_int(idx_1), ctypes.c_int(idx_2), ctypes.c_int(idx_chain))