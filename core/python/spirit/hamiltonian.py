"""
Hamiltonian
====================

Set the parameters of the Heisenberg Hamiltonian, such as external field or exchange interaction.
"""

import ctypes
from typing import NamedTuple

import numpy as np

from spirit.scalar import scalar

### Load Library
from spirit.spiritlib import _spirit

### DM vector chirality
CHIRALITY_BLOCH = 1
"""DMI Bloch chirality type for neighbour shells"""

CHIRALITY_NEEL = 2
"""DMI Neel chirality type for neighbour shells"""

CHIRALITY_BLOCH_INVERSE = -1
"""DMI Bloch chirality type for neighbour shells with opposite sign"""

CHIRALITY_NEEL_INVERSE = -2
"""DMI Neel chirality type for neighbour shells with opposite sign"""

### DDI METHOD
DDI_METHOD_NONE = 0
"""Dipole-dipole interaction: do not calculate"""

DDI_METHOD_FFT = 1
"""Dipole-dipole interaction: use FFT convolutions"""

DDI_METHOD_FMM = 2
"""Dipole-dipole interaction: use a fast multipole method (FMM)"""

DDI_METHOD_CUTOFF = 3
"""Dipole-dipole interaction: use a direct summation with a cutoff radius"""

# -------------------------------------------------------------------------------------
# ----------------------------------------- Set ---------------------------------------
# -------------------------------------------------------------------------------------

_Set_Boundary_Conditions = _spirit.Hamiltonian_Set_Boundary_Conditions
_Set_Boundary_Conditions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_bool),
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Boundary_Conditions.restype = None


def set_boundary_conditions(p_state, boundaries, idx_image=-1, idx_chain=-1):
    """Set the boundary conditions along the translation directions [a, b, c].

    0 = open, 1 = periodical
    """
    bool3 = ctypes.c_bool * 3
    _Set_Boundary_Conditions(
        ctypes.c_void_p(p_state),
        bool3(*boundaries),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# ----- External Field (Zeeman) -------------------------------------------------------


_Set_Field = _spirit.Hamiltonian_Set_Field
_Set_Field.argtypes = [
    ctypes.c_void_p,
    scalar,
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Field.restype = None


def set_field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    """Set the (homogeneous) external magnetic field."""
    vec3 = scalar * 3
    _Set_Field(
        ctypes.c_void_p(p_state),
        scalar(magnitude),
        vec3(*direction),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# ----- (uniaxial) Anisotropy ---------------------------------------------------------


_Set_Anisotropy = _spirit.Hamiltonian_Set_Anisotropy
_Set_Anisotropy.argtypes = [
    ctypes.c_void_p,
    scalar,
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Anisotropy.restype = None


def set_anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    """Set the (homogeneous) magnetocrystalline anisotropy."""
    vec3 = scalar * 3
    _Set_Anisotropy(
        ctypes.c_void_p(p_state),
        scalar(magnitude),
        vec3(*direction),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# ----- Cubic Anisotropy --------------------------------------------------------------


_Set_Cubic_Anisotropy = _spirit.Hamiltonian_Set_Cubic_Anisotropy
_Set_Cubic_Anisotropy.argtypes = [
    ctypes.c_void_p,
    scalar,
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Cubic_Anisotropy.restype = None


def set_cubic_anisotropy(p_state, magnitude, idx_image=-1, idx_chain=-1):
    """Set the (homogeneous) magnetocrystalline anisotropy."""
    _Set_Cubic_Anisotropy(
        ctypes.c_void_p(p_state),
        scalar(magnitude),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# ----- Biaxial Anisotropy ------------------------------------------------------------


_Set_Biaxial_Anisotropy = _spirit.Hamiltonian_Set_Biaxial_Anisotropy
_Set_Biaxial_Anisotropy.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(scalar, ndim=1, flags="C"),
    np.ctypeslib.ndpointer(ctypes.c_uint, ndim=2, flags="C"),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Anisotropy.restype = None


def set_biaxial_anisotropy(
    p_state,
    magnitude,
    exponents,
    primary,
    secondary,
    n_terms=None,
    idx_image=-1,
    idx_chain=-1,
):
    """Set the (homogeneous) biaxial magnetocrystalline anisotropy."""
    vec3 = scalar * 3
    if n_terms is None:
        n_terms_ = min(map(len, [magnitude, exponents]))
    else:
        n_terms_ = n_terms

    _Set_Biaxial_Anisotropy(
        ctypes.c_void_p(p_state),
        np.asarray(magnitude, dtype=scalar),
        np.asarray(exponents, dtype=ctypes.c_uint),
        vec3(*primary),
        vec3(*secondary),
        ctypes.c_int(n_terms_),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# ----- Exchange ----------------------------------------------------------------------


_Set_Exchange = _spirit.Hamiltonian_Set_Exchange
_Set_Exchange.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Exchange.restype = None


def set_exchange(p_state, n_shells, J_ij, idx_image=-1, idx_chain=-1):
    """Set the Exchange interaction in terms of neighbour shells."""
    vec = scalar * n_shells
    _Set_Exchange(
        ctypes.c_void_p(p_state),
        ctypes.c_int(n_shells),
        vec(*J_ij),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Set_DMI = _spirit.Hamiltonian_Set_DMI
_Set_DMI.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_Set_DMI.restype = None


# ----- DMI ---------------------------------------------------------------------------


def set_dmi(
    p_state, n_shells, D_ij, chirality=CHIRALITY_BLOCH, idx_image=-1, idx_chain=-1
):
    """Set the Dzyaloshinskii-Moriya interaction in terms of neighbour shells."""
    vec = scalar * n_shells
    _Set_DMI(
        ctypes.c_void_p(p_state),
        ctypes.c_int(n_shells),
        vec(*D_ij),
        ctypes.c_int(chirality),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Set_DDI = _spirit.Hamiltonian_Set_DDI
_Set_DDI.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Set_DDI.restype = None


# ----- DDI ---------------------------------------------------------------------------


def set_ddi(
    p_state,
    ddi_method,
    n_periodic_images=[4, 4, 4],
    radius=0.0,
    pb_zero_padding=True,
    idx_image=-1,
    idx_chain=-1,
):
    """Set the dipolar interaction calculation method.

    - `ddi_method`: one of the integers defined above
    - `n_periodic_images`: the number of periodical images in the three translation directions,
      taken into account when boundaries in the corresponding direction are periodical
    - `radius`: the cutoff radius for the direct summation method
    - `pb_zero_padding`: if `True` zero padding is used for periodical directions
    """
    vec3 = ctypes.c_int * 3
    _Set_DDI(
        ctypes.c_void_p(p_state),
        ctypes.c_int(ddi_method),
        vec3(*n_periodic_images),
        scalar(radius),
        ctypes.c_bool(pb_zero_padding),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# -------------------------------------------------------------------------------------
# ----------------------------------------- Get ---------------------------------------
# -------------------------------------------------------------------------------------

_Get_Name = _spirit.Hamiltonian_Get_Name
_Get_Name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Name.restype = ctypes.c_char_p


def get_name(p_state, idx_image=-1, idx_chain=-1):
    """Returns a string containing the name of the Hamiltonian currently in use."""
    return str(
        _Get_Name(
            ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)
        )
    )


_Get_Boundary_Conditions = _spirit.Hamiltonian_Get_Boundary_Conditions
_Get_Boundary_Conditions.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_bool),
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Boundary_Conditions.restype = None


def get_boundary_conditions(p_state, idx_image=-1, idx_chain=-1):
    """Returns an array of `shape(3)` containing the boundary conditions in the
    three translation directions `[a, b, c]` of the lattice.
    """
    boundaries = (3 * ctypes.c_bool)()
    _Get_Boundary_Conditions(
        ctypes.c_void_p(p_state),
        boundaries,
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )
    return [bc for bc in boundaries]


_Get_Field = _spirit.Hamiltonian_Get_Field
_Get_Field.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Field.restype = None


def get_field(p_state, idx_image=-1, idx_chain=-1):
    """Returns the magnitude and an array of `shape(3)` containing the direction of
    the external magnetic field.
    """
    magnitude = scalar()
    normal = (3 * scalar)()
    _Get_Field(
        ctypes.c_void_p(p_state),
        ctypes.byref(magnitude),
        normal,
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )
    return float(magnitude.value), [n for n in normal]


_Get_DDI = _spirit.Hamiltonian_Get_DDI
_Get_DDI.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(scalar),
    ctypes.POINTER(ctypes.c_bool),
    ctypes.c_int,
    ctypes.c_int,
]
_Get_DDI.restype = None


# ----- DDI ---------------------------------------------------------------------------


def get_ddi(p_state, idx_image=-1, idx_chain=-1):
    """Returns a dictionary, containing information about the used ddi settings"""
    ddi_method = ctypes.c_int()
    n_periodic_images = (3 * ctypes.c_int)()
    cutoff_radius = scalar()
    pb_zero_padding = ctypes.c_bool()
    _Get_DDI(
        ctypes.c_void_p(p_state),
        ctypes.byref(ddi_method),
        n_periodic_images,
        ctypes.byref(cutoff_radius),
        ctypes.byref(pb_zero_padding),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )
    return {
        "ddi_method": ddi_method.value,
        "n_periodic_images": [i for i in n_periodic_images],
        "cutoff_radius": cutoff_radius.value,
        "pb_zero_padding": pb_zero_padding.value,
    }


# ----- Hessian -----------------------------------------------------------------------


_Write_Hessian = _spirit.Hamiltonian_Write_Hessian
_Write_Hessian.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Write_Hessian.restype = None


def write_hessian(p_state, filename, triplet_format=True, idx_image=-1, idx_chain=-1):
    """RWrites the embedding Hessian to a file"""
    _Write_Hessian(
        ctypes.c_void_p(p_state),
        ctypes.c_char_p(filename.encode("utf-8")),
        ctypes.c_bool(triplet_format),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


# ----- Anisotropy --------------------------------------------------------------------


_Get_Anisotropy = _spirit.Hamiltonian_Get_Anisotropy
_Get_Anisotropy.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Anisotropy.restype = None


def get_anisotropy(p_state, idx_image=-1, idx_chain=-1):
    """Get the magnetocrystalline anisotropy."""
    magnitude = scalar()
    normal = (3 * scalar)()
    _Get_Anisotropy(
        ctypes.c_void_p(p_state),
        ctypes.byref(magnitude),
        normal,
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )
    return float(magnitude.value), [n for n in normal]


# ----- Cubic Anisotropy --------------------------------------------------------------


_Get_Cubic_Anisotropy = _spirit.Hamiltonian_Get_Cubic_Anisotropy
_Get_Cubic_Anisotropy.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Cubic_Anisotropy.restype = None


def get_cubic_anisotropy(p_state, idx_image=-1, idx_chain=-1):
    """Set the cubic magnetocrystalline anisotropy."""
    magnitude = scalar()
    _Get_Cubic_Anisotropy(
        ctypes.c_void_p(p_state),
        ctypes.byref(magnitude),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )
    return float(magnitude.value)


# ----- Biaxial Anisotropy ------------------------------------------------------------


_Get_Biaxial_Anisotropy_N_Atoms = _spirit.Hamiltonian_Get_Biaxial_Anisotropy_N_Atoms
_Get_Biaxial_Anisotropy_N_Atoms.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Biaxial_Anisotropy_N_Atoms.restype = ctypes.c_int


def get_biaxial_anisotropy_n_atoms(p_state, idx_image=-1, idx_chain=-1):
    """Get the number of atoms for which a biaxial anisotropy is defined."""
    result = _Get_Biaxial_Anisotropy_N_Atoms(
        ctypes.c_void_p(p_state),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )

    return int(result)


_Get_Biaxial_Anisotropy_N_Terms = _spirit.Hamiltonian_Get_Biaxial_Anisotropy_N_Terms
_Get_Biaxial_Anisotropy_N_Terms.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Biaxial_Anisotropy_N_Terms.restype = ctypes.c_int


def get_biaxial_anisotropy_n_terms(p_state, idx_image=-1, idx_chain=-1):
    """Get the number of terms that contribute to the biaxial anisotropy."""

    result = _Get_Biaxial_Anisotropy_N_Terms(
        ctypes.c_void_p(p_state),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )

    return int(result)


class BiaxialAnisotropyData(NamedTuple):
    indices: np.ndarray
    primary: np.ndarray
    secondary: np.ndarray
    site_p: np.ndarray
    magnitude: np.ndarray
    exponents: np.ndarray


_Get_Biaxial_Anisotropy = _spirit.Hamiltonian_Get_Biaxial_Anisotropy
_Get_Biaxial_Anisotropy.argtypes = [
    ctypes.c_void_p,
    np.ctypeslib.ndpointer(ctypes.c_int, ndim=1, flags=["C", "W"]),
    np.ctypeslib.ndpointer(scalar, ndim=2, flags=["C", "W"]),
    np.ctypeslib.ndpointer(scalar, ndim=2, flags=["C", "W"]),
    np.ctypeslib.ndpointer(ctypes.c_uint, ndim=1, flags=["C", "W"]),
    ctypes.c_int,
    np.ctypeslib.ndpointer(scalar, ndim=1, flags=["C", "W"]),
    np.ctypeslib.ndpointer(ctypes.c_uint, ndim=2, flags=["C", "W"]),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
_Get_Biaxial_Anisotropy.restype = None


def get_biaxial_anisotropy(
    p_state, n_indices=None, n_terms=None, idx_image=-1, idx_chain=-1
):
    """Get the data representing the biaxial anisotropy."""
    if n_indices is None:
        n_indices_ = get_biaxial_anisotropy_n_atoms(
            p_state, idx_image=idx_image, idx_chain=idx_chain
        )
    else:
        n_indices_ = n_indices

    if n_terms is None:
        n_terms_ = get_biaxial_anisotropy_n_terms(
            p_state, idx_image=idx_image, idx_chain=idx_chain
        )
    else:
        n_terms_ = n_terms

    indices = np.zeros((n_indices_,), dtype=ctypes.c_int)
    primary = np.zeros((n_indices_, 3), dtype=scalar)
    secondary = np.zeros((n_indices_, 3), dtype=scalar)
    site_p = np.zeros((n_indices_ + 1,), dtype=ctypes.c_uint)
    magnitude = np.zeros((n_terms_,), dtype=scalar)
    exponents = np.zeros((n_terms_, 3), dtype=ctypes.c_uint)

    _Get_Biaxial_Anisotropy(
        ctypes.c_void_p(p_state),
        indices,
        primary,
        secondary,
        site_p,
        ctypes.c_int(n_indices_),
        magnitude,
        exponents,
        ctypes.c_int(n_terms_),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )

    return BiaxialAnisotropyData(
        indices, primary, secondary, site_p, magnitude, exponents
    )
