"""
Hamiltonian
====================

Set the parameters of the Heisenberg Hamiltonian, such as external field or exchange interaction.
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

### DM vector chirality
CHIRALITY_BLOCH         =  1
"""DMI Bloch chirality type for neighbour shells"""

CHIRALITY_NEEL          =  2
"""DMI Neel chirality type for neighbour shells"""

CHIRALITY_BLOCH_INVERSE = -1
"""DMI Bloch chirality type for neighbour shells with opposite sign"""

CHIRALITY_NEEL_INVERSE  = -2
"""DMI Neel chirality type for neighbour shells with opposite sign"""

### DDI METHOD
DDI_METHOD_NONE         = 0
"""Dipole-dipole interaction: do not calculate"""

DDI_METHOD_FFT          = 1
"""Dipole-dipole interaction: use FFT convolutions"""

DDI_METHOD_FMM          = 2
"""Dipole-dipole interaction: use a fast multipole method (FMM)"""

DDI_METHOD_CUTOFF       = 3
"""Dipole-dipole interaction: use a direct summation with a cutoff radius"""

### ---------------------------------- Set ----------------------------------

_Set_Boundary_Conditions             = _spirit.Hamiltonian_Set_Boundary_Conditions
_Set_Boundary_Conditions.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool),
                          ctypes.c_int, ctypes.c_int]
_Set_Boundary_Conditions.restype     = None
def set_boundary_conditions(p_state, boundaries, idx_image=-1, idx_chain=-1):
    """Set the boundary conditions along the translation directions [a, b, c].

    0 = open, 1 = periodical
    """
    bool3 = ctypes.c_bool * 3
    _Set_Boundary_Conditions(ctypes.c_void_p(p_state), bool3(*boundaries),
                             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_Field             = _spirit.Hamiltonian_Set_Field
_Set_Field.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int, ctypes.c_int]
_Set_Field.restype     = None
def set_field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    """Set the (homogeneous) external magnetic field."""
    vec3 = ctypes.c_float * 3
    _Set_Field(ctypes.c_void_p(p_state), ctypes.c_float(magnitude), vec3(*direction),
               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_Anisotropy             = _spirit.Hamiltonian_Set_Anisotropy
_Set_Anisotropy.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Set_Anisotropy.restype     = None
def set_anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    """Set the (homogeneous) magnetocrystalline anisotropy."""
    vec3 = ctypes.c_float * 3
    _Set_Anisotropy(ctypes.c_void_p(p_state), ctypes.c_float(magnitude), vec3(*direction),
                    ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_Exchange             = _spirit.Hamiltonian_Set_Exchange
_Set_Exchange.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Set_Exchange.restype     = None
def set_exchange(p_state, n_shells, J_ij, idx_image=-1, idx_chain=-1):
    """Set the Exchange interaction in terms of neighbour shells."""
    vec = ctypes.c_float * n_shells
    _Set_Exchange(ctypes.c_void_p(p_state), ctypes.c_int(n_shells), vec(*J_ij),
                  ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_DMI             = _spirit.Hamiltonian_Set_DMI
_Set_DMI.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_DMI.restype     = None
def set_dmi(p_state, n_shells, D_ij, chirality=CHIRALITY_BLOCH, idx_image=-1, idx_chain=-1):
    """Set the Dzyaloshinskii-Moriya interaction in terms of neighbour shells."""
    vec = ctypes.c_float * n_shells
    _Set_DMI(ctypes.c_void_p(p_state), ctypes.c_int(n_shells), vec(*D_ij),
             ctypes.c_int(chirality), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Set_DDI             = _spirit.Hamiltonian_Set_DDI
_Set_DDI.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_float,
                        ctypes.c_int, ctypes.c_int]
_Set_DDI.restype     = None
def set_ddi(p_state, ddi_method, n_periodic_images=[4,4,4], radius=0.0, idx_image=-1, idx_chain=-1):
    """Set the dipolar interaction calculation method.

    - ddi_method -- one of the integers defined above
    - n_periodic_images -- the number of periodical images in the three translation directions, taken into account
      when boundaries in the corresponding direction are periodical
    - radius -- the cutoff radius for the direct summation method
    """
    vec3 = ctypes.c_int * 3
    _Set_DDI(ctypes.c_void_p(p_state), ctypes.c_int(ddi_method) , vec3(*n_periodic_images), ctypes.c_float(radius),
             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### ---------------------------------- Get ----------------------------------

_Get_Name          = _spirit.Hamiltonian_Get_Name
_Get_Name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Name.restype  = ctypes.c_char_p
def get_name(p_state, idx_image=-1, idx_chain=-1):
    """Returns a string containing the name of the Hamiltonian currently in use."""
    return str(_Get_Name(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_Get_Boundary_Conditions          = _spirit.Hamiltonian_Get_Boundary_Conditions
_Get_Boundary_Conditions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool),
                                     ctypes.c_int, ctypes.c_int]
_Get_Boundary_Conditions.restype  = None
def get_boundary_conditions(p_state, idx_image=-1, idx_chain=-1):
    """Returns an array of shape (3) containing the boundary conditions in the
    three translation directions [a, b, c] of the lattice.
    """
    boundaries = (3*ctypes.c_bool)()
    _Get_Boundary_Conditions(ctypes.c_void_p(p_state), boundaries,
                             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [bc for bc in boundaries]

_Get_Field          = _spirit.Hamiltonian_Get_Field
_Get_Field.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Field.restype  = None
def get_field(p_state, idx_image=-1, idx_chain=-1):
    """Returns the magnitude and an array of shape (3) containing the direction of
    the external magnetic field.
    """
    magnitude = (1*ctypes.c_float)()
    normal = (3*ctypes.c_float)()
    _Get_Field(ctypes.c_void_p(p_state), magnitude, normal,
               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return float(magnitude), [n for n in normal]

_Get_DDI          = _spirit.Hamiltonian_Get_DDI
_Get_DDI.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_DDI.restype  = ctypes.c_float
def get_ddi(p_state, idx_image=-1, idx_chain=-1):
    """Returns the cutoff radius of the DDI."""
    return float(_Get_DDI(ctypes.c_void_p(p_state), ctypes.c_int(idx_image),
                          ctypes.c_int(idx_chain)))