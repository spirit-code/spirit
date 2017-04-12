import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


# The Bohr Magneton [meV / T]
_mu_B             = _spirit.Constants_mu_B
_mu_B.argtypes    = None
_mu_B.restype     = scalar
def mu_B():
    return _mu_B()

# The Boltzmann constant [meV / K]
_k_B             = _spirit.Constants_k_B
_k_B.argtypes    = None
_k_B.restype     = scalar
def k_B():
    return _k_B()