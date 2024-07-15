"""
Configuration
====================

Set various spin configurations, such as homogeneous domains, spirals or skyrmions.

All configuration setters support the following arguments with default values:

- `pos=[0,0,0]`: the centre of the configuration, relative to the centre of the system
- `border_rectangular=[-1,-1,-1]`: values > 0 mean a restriction in `+` and `-` direction relative to the position
- `border_cylindrical=-1`: restricts the initialisation to a z-aligned cylinder around the position
- `border_spherical=-1`: restricts the initialisation to a sphere around the position
- `inverted=False`: exactly inverts the above restrictions
"""

from spirit.scalar import scalar
import ctypes

### Load Library
from spirit.spiritlib import _spirit

_Domain = _spirit.Configuration_Domain
_Domain.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Domain.restype = None


def domain(
    p_state,
    dir,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set a domain (homogeneous) configuration."""
    vec3 = scalar * 3
    _Domain(
        ctypes.c_void_p(p_state),
        vec3(*dir),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_PlusZ = _spirit.Configuration_PlusZ
_PlusZ.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_PlusZ.restype = None


def plus_z(
    p_state,
    pos=[0.0, 0.0, 0.0],
    border_rectangular=[-1.0, -1.0, -1.0],
    border_cylindrical=-1.0,
    border_spherical=-1.0,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set a `+z` (homogeneous) configuration."""
    vec3 = scalar * 3
    _PlusZ(
        ctypes.c_void_p(p_state),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_MinusZ = _spirit.Configuration_MinusZ
_MinusZ.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_MinusZ.restype = None


def minus_z(
    p_state,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set a `-z` (homogeneous) configuration."""
    vec3 = scalar * 3
    _MinusZ(
        ctypes.c_void_p(p_state),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Random = _spirit.Configuration_Random
_Random.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Random.restype = None


def random(
    p_state,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Distribute all spins randomly on the unit sphere."""
    vec3 = scalar * 3
    _Random(
        ctypes.c_void_p(p_state),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        False,
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Add_Noise_Temperature = _spirit.Configuration_Add_Noise_Temperature
_Add_Noise_Temperature.argtypes = [
    ctypes.c_void_p,
    scalar,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Add_Noise_Temperature.restype = None


def add_noise(
    p_state,
    temperature,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Add temperature-scaled random noise to configuration."""
    vec3 = scalar * 3
    _Add_Noise_Temperature(
        ctypes.c_void_p(p_state),
        scalar(temperature),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Skyrmion = _spirit.Configuration_Skyrmion
_Skyrmion.argtypes = [
    ctypes.c_void_p,
    scalar,
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Skyrmion.restype = None


def skyrmion(
    p_state,
    radius,
    order=1,
    phase=1,
    up_down=False,
    achiral=False,
    right_left=False,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set a skyrmion configuration.

    Arguments:

    - `radius`: the extent of the skyrmion, at which it points approximately upwards. The skyrmion only extends up to `radius`, meaning that `border_cylindrical` is not usually necessary.

    Keyword arguments:

    - `order`: the number of twists along a circle cutting the skyrmion
    - `phase`: 0 corresponds to a Neel skyrmion, -90 to a Bloch skyrmion
    - `up_down`: if `True`, the z-orientation is inverted
    - `achiral`: if `True`, the topological charge is inverted
    - `right_left`: if `True`, the in-plane rotation is inverted

    """
    vec3 = scalar * 3
    _Skyrmion(
        ctypes.c_void_p(p_state),
        scalar(radius),
        scalar(order),
        scalar(phase),
        ctypes.c_bool(up_down),
        ctypes.c_bool(achiral),
        ctypes.c_bool(right_left),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_DW_Skyrmion = _spirit.Configuration_DW_Skyrmion
_DW_Skyrmion.argtypes = [
    ctypes.c_void_p,
    scalar,
    scalar,
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_DW_Skyrmion.restype = None


def dw_skyrmion(
    p_state,
    dw_radius,
    dw_width,
    order=1,
    phase=1,
    up_down=False,
    achiral=False,
    right_left=False,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set a 360 degree domain wall skyrmion configuration.

    Arguments:

    - `dw_radius`: the radius of the circular domain wall skyrmion.
    - `dw_width`: the width of the domain wall circumference of the skyrmion.

    Keyword arguments:

    - `order`: the number of twists along a circle cutting the skyrmion
    - `phase`: 0 corresponds to a Neel skyrmion, -90 to a Bloch skyrmion
    - `up_down`: if `True`, the z-orientation is inverted
    - `achiral`: if `True`, the topological charge is inverted
    - `right_left`: if `True`, the in-plane rotation is inverted

    """
    vec3 = scalar * 3
    _DW_Skyrmion(
        ctypes.c_void_p(p_state),
        scalar(dw_radius),
        scalar(dw_width),
        scalar(order),
        scalar(phase),
        ctypes.c_bool(up_down),
        ctypes.c_bool(achiral),
        ctypes.c_bool(right_left),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Hopfion = _spirit.Configuration_Hopfion
_Hopfion.argtypes = [
    ctypes.c_void_p,
    scalar,
    ctypes.c_int,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.POINTER(scalar),
    ctypes.c_int,
    ctypes.c_int,
]
_Hopfion.restype = None


def hopfion(
    p_state,
    radius,
    order=1,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    normal=[0, 0, 1],
    idx_image=-1,
    idx_chain=-1,
):
    """Set a Hopfion configuration.

    Arguments:

    - `radius`: the distance from the center to the center of the corresponding tubular isosurface

    Keyword arguments:

    - `order`: the number of windings of the toroidal hopfion

    In contrast to the skyrmion, it extends over the whole allowed space.
    """
    vec3 = scalar * 3
    _Hopfion(
        ctypes.c_void_p(p_state),
        scalar(radius),
        ctypes.c_int(order),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        vec3(*normal),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_SpinSpiral = _spirit.Configuration_SpinSpiral
_SpinSpiral.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_SpinSpiral.restype = None


def spin_spiral(
    p_state,
    direction_type,
    q_vector,
    axis,
    theta,
    pos=[0, 0, 0],
    border_rectangular=[-1, -1, -1],
    border_cylindrical=-1,
    border_spherical=-1,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set a spin spiral configuration.

    TODO: document parameters

    - `direction_type`:
    - `q_vector`:
    - `axis`:
    - `theta`:
    """
    vec3 = scalar * 3
    _SpinSpiral(
        ctypes.c_void_p(p_state),
        ctypes.c_char_p(direction_type.encode("utf-8")),
        vec3(*q_vector),
        vec3(*axis),
        scalar(theta),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Set_Pinned = _spirit.Configuration_Set_Pinned
_Set_Pinned.argtypes = [
    ctypes.c_void_p,
    ctypes.c_bool,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Pinned.restype = None


def set_pinned(
    p_state,
    pinned,
    pos=[0.0, 0.0, 0.0],
    border_rectangular=[-1.0, -1.0, -1.0],
    border_cylindrical=-1.0,
    border_spherical=-1.0,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set whether the spins within the given region are pinned or not.

    If they are pinned, they are pinned to their current orientation.
    """
    vec3 = scalar * 3
    _Set_Pinned(
        ctypes.c_void_p(p_state),
        ctypes.c_bool(pinned),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )


_Set_Atom_Type = _spirit.Configuration_Set_Atom_Type
_Set_Atom_Type.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(scalar),
    ctypes.POINTER(scalar),
    scalar,
    scalar,
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.c_int,
]
_Set_Atom_Type.restype = None


def set_atom_type(
    p_state,
    atom_type=0,
    pos=[0.0, 0.0, 0.0],
    border_rectangular=[-1.0, -1.0, -1.0],
    border_cylindrical=-1.0,
    border_spherical=-1.0,
    inverted=False,
    idx_image=-1,
    idx_chain=-1,
):
    """Set the type of the atoms in the given region (default: 0).

    This can be used e.g. to insert defects (-1).
    """
    vec3 = scalar * 3
    _Set_Atom_Type(
        ctypes.c_void_p(p_state),
        ctypes.c_int(atom_type),
        vec3(*pos),
        vec3(*border_rectangular),
        scalar(border_cylindrical),
        scalar(border_spherical),
        ctypes.c_bool(inverted),
        ctypes.c_int(idx_image),
        ctypes.c_int(idx_chain),
    )
