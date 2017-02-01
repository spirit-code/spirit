import core.corelib as corelib
from core.scalar import scalar
import ctypes

### Load Library
_core = corelib.LoadCoreLibrary()


# The Bohr Magneton [meV / T]
_mu_B             = _core.Constants_mu_B
_mu_B.argtypes    = None
_mu_B.restype     = scalar
def mu_B():
    return _mu_B()

# The Boltzmann constant [meV / K]
_k_B             = _core.Constants_k_B
_k_B.argtypes    = None
_k_B.restype     = scalar
def k_B():
    return _k_B()