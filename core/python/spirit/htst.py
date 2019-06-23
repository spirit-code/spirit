"""
HTST
====================

Harmonic transition state theory.

Note that `calculate_prefactor` needs to be called before using any of the getter functions.
"""

import spirit.spiritlib as spiritlib
import spirit.parameters as parameters
import spirit.system as system
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()


_Calculate          = _spirit.HTST_Calculate
_Calculate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Calculate.restype  = ctypes.c_float
def calculate(p_state, idx_image_minimum, idx_image_sp, idx_chain=-1):
    """Performs an HTST calculation and returns rate prefactor.

    *Note:* this function must be called before any of the getters.
    """
    return _Calculate(p_state, idx_image_minimum, idx_image_sp, idx_chain)


### Get HTST transition rate components
_Get_Info          = _spirit.HTST_Get_Info
_Get_Info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Info.restype  = None
def get_info(p_state, idx_chain=-1):
    """Returns a set of HTST information:

    - the exponent of the temperature-dependence
    - `me`
    - `Omega_0`
    - `s`
    - zero mode volume at the minimum
    - zero mode volume at the saddle point
    - dynamical prefactor
    - full rate prefactor (without temperature dependent part)
    """
    temperature_exponent = ctypes.c_float()
    me                   = ctypes.c_float()
    Omega_0              = ctypes.c_float()
    s                    = ctypes.c_float()
    volume_min           = ctypes.c_float()
    volume_sp            = ctypes.c_float()
    prefactor_dynamical  = ctypes.c_float()
    prefactor            = ctypes.c_float()

    _Get_Info(ctypes.c_void_p(p_state), ctypes.pointer(temperature_exponent), ctypes.pointer(me),
                ctypes.pointer(Omega_0), ctypes.pointer(s), ctypes.pointer(volume_min),
                ctypes.pointer(volume_sp), ctypes.pointer(prefactor_dynamical),
                ctypes.pointer(prefactor), ctypes.c_int(idx_chain))

    return temperature_exponent.value, me.value, Omega_0.value, s.value,\
            volume_min.value, volume_sp.value, prefactor_dynamical.value, prefactor.value


_Get_Eigenvalues_Min          = _spirit.HTST_Get_Eigenvalues_Min
_Get_Eigenvalues_Min.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Eigenvalues_Min.restype  = None
def get_eigenvalues_min(p_state, idx_chain=-1):
    """Returns the eigenvalues at the minimum.
    Shape (2*nos)
    """
    nos             = system.get_nos(p_state, -1, idx_chain)
    eigenvalues_min = (2*nos*ctypes.c_float)()
    _Get_Eigenvalues_Min(ctypes.c_void_p(p_state), eigenvalues_min, ctypes.c_int(idx_chain))
    return eigenvalues_min


_Get_Eigenvectors_Min          = _spirit.HTST_Get_Eigenvectors_Min
_Get_Eigenvectors_Min.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Eigenvectors_Min.restype  = None
def get_eigenvectors_min(p_state, idx_chain=-1):
    """Returns the eigenvectors at the minimum.
    Shape (2*nos*nos)
    """
    nos              = system.get_nos(p_state, -1, idx_chain)
    eigenvectors_min = (2*nos*nos*ctypes.c_float)()
    _Get_Eigenvectors_Min(ctypes.c_void_p(p_state), eigenvectors_min, ctypes.c_int(idx_chain))
    return eigenvectors_min


_Get_Eigenvalues_SP          = _spirit.HTST_Get_Eigenvalues_SP
_Get_Eigenvalues_SP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Eigenvalues_SP.restype  = None
def get_eigenvalues_sp(p_state, idx_chain=-1):
    """Returns the eigenvalues at the saddle point.
    Shape (2*nos)
    """
    nos            = system.get_nos(p_state, -1, idx_chain)
    eigenvalues_sp = (2*nos*ctypes.c_float)()
    _Get_Eigenvalues_SP(ctypes.c_void_p(p_state), eigenvalues_sp, ctypes.c_int(idx_chain))
    return eigenvalues_sp


_Get_Eigenvectors_SP          = _spirit.HTST_Get_Eigenvectors_SP
_Get_Eigenvectors_SP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Eigenvectors_SP.restype  = None
def get_eigenvectors_sp(p_state, idx_chain=-1):
    """Returns the eigenvectors at the saddle point.
    Shape (2*nos*nos)
    """
    nos                 = system.get_nos(p_state, -1, idx_chain)
    eigenvectors_sp     = (2*nos*nos*ctypes.c_float)()
    _Get_Eigenvectors_SP(ctypes.c_void_p(p_state), eigenvectors_sp, ctypes.c_int(idx_chain))
    return eigenvectors_sp


_Get_Velocities          = _spirit.HTST_Get_Velocities
_Get_Velocities.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_Get_Velocities.restype  = None
def get_velocities(p_state, idx_chain=-1):
    """Returns the velocities perpendicular to the dividing surface.
    Shape (2*nos)
    """
    nos        = system.get_nos(p_state, -1, idx_chain)
    velocities = (2*nos*ctypes.c_float)()
    _Get_Velocities(ctypes.c_void_p(p_state), velocities, ctypes.c_int(idx_chain))
    return velocities