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
_Set_DDI.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_float, ctypes.c_bool,
                        ctypes.c_int, ctypes.c_int]
_Set_DDI.restype     = None
def set_ddi(p_state, ddi_method, n_periodic_images=[4,4,4], radius=0.0, pb_zero_padding=True, idx_image=-1, idx_chain=-1):
    """Set the dipolar interaction calculation method.

    - `ddi_method`: one of the integers defined above
    - `n_periodic_images`: the number of periodical images in the three translation directions,
      taken into account when boundaries in the corresponding direction are periodical
    - `radius`: the cutoff radius for the direct summation method
    - `pb_zero_padding`: if `True` zero padding is used for periodical directions
    """
    vec3 = ctypes.c_int * 3
    _Set_DDI(ctypes.c_void_p(p_state), ctypes.c_int(ddi_method) , vec3(*n_periodic_images), ctypes.c_float(radius), ctypes.c_bool(pb_zero_padding),
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
    """Returns an array of `shape(3)` containing the boundary conditions in the
    three translation directions `[a, b, c]` of the lattice.
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
    """Returns the magnitude and an array of `shape(3)` containing the direction of
    the external magnetic field.
    """
    magnitude = ctypes.c_float()
    normal = (3*ctypes.c_float)()
    _Get_Field(ctypes.c_void_p(p_state), ctypes.byref(magnitude), normal,
               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return float(magnitude.value), [n for n in normal]

_Get_DDI          = _spirit.Hamiltonian_Get_DDI
_Get_DDI.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_bool), ctypes.c_int, ctypes.c_int]
_Get_DDI.restype  = None
def get_ddi(p_state, idx_image=-1, idx_chain=-1):
    """Returns a dictionary, containing information about the used ddi settings"""
    ddi_method = ctypes.c_int()
    n_periodic_images = (3*ctypes.c_int)()
    cutoff_radius = ctypes.c_float()
    pb_zero_padding = ctypes.c_bool()
    _Get_DDI(ctypes.c_void_p(p_state), ctypes.byref(ddi_method), n_periodic_images, ctypes.byref(cutoff_radius), ctypes.byref(pb_zero_padding), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return { "ddi_method" : ddi_method.value,
            "n_periodic_images" : [ i for i in n_periodic_images ],
            "cutoff_radius" : cutoff_radius.value,
            "pb_zero_padding" : pb_zero_padding.value }

_Get_Anisotropy = _spirit.Hamiltonian_Get_Anisotropy
_Get_Anisotropy.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_float)
                            , ctypes.POINTER(ctypes.c_float)
                            , ctypes.c_int, ctypes.c_int]
_Get_Anisotropy.restype = None
def get_anisotropy(p_state, idx_image=-1, idx_chain=-1):
    """Returns the magnitude and an array of `shape(3)` containing the direction of
    the (homogeneous) magnetocrystalline anisotropy.
    """
    magnitude = ctypes.c_float()
    normal = (3*ctypes.c_float)()
    _Get_Anisotropy(ctypes.c_void_p(p_state), ctypes.byref(magnitude), normal
                    ,ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return float(magnitude.value), [n for n in normal]

_Get_Exchange_Shells = _spirit.Hamiltonian_Get_Exchange_Shells
_Get_Exchange_Shells.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)
                            , ctypes.POINTER(ctypes.c_float)
                            , ctypes.c_int, ctypes.c_int]
_Get_Exchange_Shells.restype = None
def get_Exchange_shells(p_state, idx_image=-1, idx_chain=-1):
    """Returns the magnitude and an array of `n_shells` containing the direction of
    the Exchange interaction.
    """
    NULL = ctypes.POINTER(ctypes.c_float)()
    #Null = None
    n_shells = ctypes.c_int()
    #Only write n_shells and chirality
    _Get_Exchange_Shells(ctypes.c_void_p(p_state),ctypes.byref(n_shells),NULL
             ,ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    jij = (n_shells.value*ctypes.c_float)()
    _Get_Exchange_Shells(ctypes.c_void_p(p_state),ctypes.byref(n_shells),jij
             ,ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_shells.value), [float(j) for j in jij]

_Get_DMI_Shells = _spirit.Hamiltonian_Get_DMI_Shells
_Get_DMI_Shells.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)
                            , ctypes.POINTER(ctypes.c_float)
                            , ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
_Get_DMI_Shells.restype = None
def get_DMI_shells(p_state, idx_image=-1, idx_chain=-1):
    """Returns the magnitude and an array of `n_shells` containing the direction of
    the DMI interaction.
    """
    NULL = ctypes.POINTER(ctypes.c_float)()
    #Null = None
    n_shells = ctypes.c_int()
    chirality = ctypes.c_int()
    #Only write n_shells and chirality
    _Get_DMI_Shells(ctypes.c_void_p(p_state),ctypes.byref(n_shells),NULL
             ,ctypes.byref(chirality),ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    dij = (n_shells.value*ctypes.c_float)()
    _Get_DMI_Shells(ctypes.c_void_p(p_state),ctypes.byref(n_shells),dij
             ,ctypes.byref(chirality),ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return int(n_shells.value),[float(d) for d in dij], int(chirality.value)
_Write_Hessian          = _spirit.Hamiltonian_Write_Hessian
_Write_Hessian.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_int, ctypes.c_int]
_Write_Hessian.restype  = None
def write_hessian(p_state, filename, triplet_format=True, idx_image=-1, idx_chain=-1):
    """RWrites the embedding Hessian to a file"""
    _Write_Hessian(ctypes.c_void_p(p_state), ctypes.c_char_p(filename.encode('utf-8')), ctypes.c_bool(triplet_format), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
