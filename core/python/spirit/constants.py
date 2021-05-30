"""
Constants
====================
"""

import spirit.spiritlib as spiritlib
import ctypes

from spirit.scalar import scalar

# Load Library
_spirit = spiritlib.load_spirit_library()

_mu_B           = _spirit.Constants_mu_B
_mu_B.argtypes  = None
_mu_B.restype   = scalar
mu_B = _mu_B()
"""The Bohr Magneton [meV / T]"""

_mu_0           = _spirit.Constants_mu_0
_mu_0.argtypes  = None
_mu_0.restype   = scalar
mu_0 = _mu_0()
"""The vacuum permeability [T^2 m^3 / meV]"""

_k_B            = _spirit.Constants_k_B
_k_B.argtypes   = None
_k_B.restype    = scalar
k_B = _k_B()
"""The Boltzmann constant [meV / K]"""

_hbar           = _spirit.Constants_hbar
_hbar.argtypes  = None
_hbar.restype   = scalar
hbar = _hbar()
"""Planck's constant [meV*ps / rad]"""

_mRy            = _spirit.Constants_mRy
_mRy.argtypes   = None
_mRy.restype    = scalar
mRy = _mRy()
"""Millirydberg [mRy / meV]"""

_gamma          = _spirit.Constants_gamma
_gamma.argtypes = None
_gamma.restype  = scalar
gamma = _gamma()
"""Gyromagnetic ratio of electron [rad / (ps*T)]"""

_g_e            = _spirit.Constants_g_e
_g_e.argtypes   = None
_g_e.restype    = scalar
g_e = _g_e()
"""Electron g-factor [unitless]"""

_Pi            = _spirit.Constants_Pi
_Pi.argtypes   = None
_Pi.restype    = scalar
pi = _Pi()
"""Pi [rad]"""