"""
Quantities
====================
"""

import spirit.spiritlib as spiritlib
import spirit.system as system
from spirit.scalar import scalar
import ctypes

import numpy as np

### Load Library
_spirit = spiritlib.load_spirit_library()

_Get_Magnetization          = _spirit.Quantity_Get_Magnetization
_Get_Magnetization.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Get_Magnetization.restype  = None
def get_magnetization(p_state, idx_image=-1, idx_chain=-1):
    """Calculates and returns the average magnetization of the system as
    an array of shape (3).
    """
    magnetization = (3*ctypes.c_float)()
    _Get_Magnetization(ctypes.c_void_p(p_state), magnetization,
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [float(i) for i in magnetization]


### Temporary info function for MMF
_Get_MinimumMode            = _spirit.Quantity_Get_Grad_Force_MinimumMode
_Get_MinimumMode.argtypes   = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_MinimumMode.restype    = None
def get_mmf_info(p_state, idx_image=-1, idx_chain=-1):
    """Returns a set of MMF information, meant mostly for testing or debugging.

    - numpy.array_view of shape (NOS, 3) of the energy gradient
    - the lowest eigenvalue
    - numpy.array_view of shape (NOS, 3) of the eigenmode
    - numpy.array_view of shape (NOS, 3) of the force
    """
    nos = system.get_nos(p_state, idx_image, idx_chain)

    ArrayType = ctypes.c_float*(3*nos)

    MM = []*(3*nos)
    _MM = ArrayType(*MM)
    FF = []*(3*nos)
    _FF = ArrayType(*FF)
    GG = []*(3*nos)
    _GG = ArrayType(*FF)
    ev = []*1
    _eval = ArrayType(*ev)

    _Get_MinimumMode(p_state, _GG, _eval, _MM, _FF, idx_image, idx_chain)

    # array_pointer = ctypes.cast(_MM, ctypes.POINTER(ArrayType))
    # array = np.frombuffer(array_pointer.contents)

    MMM = np.array(_MM)
    FFF = np.array(_FF)
    GGG = np.array(_GG)
    array_view_mode = MMM.view()
    array_view_mode.shape = (nos, 3)
    array_view_force = FFF.view()
    array_view_force.shape = (nos, 3)
    array_view_grad = GGG.view()
    array_view_grad.shape = (nos, 3)

    return array_view_grad, _eval[0], array_view_mode, array_view_force


_Get_Topological_Charge          = _spirit.Quantity_Get_Topological_Charge
_Get_Topological_Charge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Topological_Charge.restype  = ctypes.c_float
def get_topological_charge(p_state, idx_image=-1, idx_chain=-1):
    """Calculates and returns the total topological charge of 2D systems.

    Note that the charge can take unphysical non-integer values for open boundaries,
    because it is not well-defined in this case.

    Returns 0 for systems of other dimensionality.
    """
    return float(_Get_Topological_Charge(ctypes.c_void_p(p_state),
                       ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))