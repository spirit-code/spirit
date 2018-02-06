import spirit.spiritlib as spiritlib
import ctypes

from spirit.scalar import scalar

# Load Library
_spirit = spiritlib.LoadSpiritLibrary()

# The Bohr Magneton [meV / T]
_mu_B           = _spirit.Constants_mu_B
_mu_B.argtypes  = None
_mu_B.restype   = scalar
def mu_B():
    return _mu_B()

# The vacuum permeability [T^2 m^3 / meV]
_mu_0           = _spirit.Constants_mu_0
_mu_0.argtypes  = None
_mu_0.restype   = scalar
def mu_0():
    return _mu_0()

# The Boltzmann constant [meV / K]
_k_B            = _spirit.Constants_k_B
_k_B.argtypes   = None
_k_B.restype    = scalar
def k_B():
    return _k_B()

# Planck's constant [meV*ps / rad]
_hbar           = _spirit.Constants_hbar
_hbar.argtypes  = None
_hbar.restype   = scalar
def hbar():
    return _hbar()

# Millirydberg [mRy / meV]
_mRy            = _spirit.Constants_mRy
_mRy.argtypes   = None
_mRy.restype    = scalar
def mRy():
    return _mRy()

# Gyromagnetic ratio of electron [rad / (ps*T)]
_gamma          = _spirit.Constants_gamma
_gamma.argtypes = None
_gamma.restype  = scalar
def gamma():
    return _gamma()

# Electron g-factor [unitless]
_g_e            = _spirit.Constants_g_e
_g_e.argtypes   = None
_g_e.restype    = scalar
def g_e():
    return _g_e()

# Pi [rad]
_Pi            = _spirit.Constants_Pi
_Pi.argtypes   = None
_Pi.restype    = scalar
def Pi():
    return _Pi()