"""
Legacy
====================

The legacy module contains functionalty that is provided for backwards compatibility.

This functionalty may be dropped in future releases and should be considered deprecated.
"""

import ctypes

### Load Library
from spirit.spiritlib import _spirit

_Convert_Config_to_TOML = _spirit.Legacy_Convert_Config_to_TOML
_Convert_Config_to_TOML.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_Convert_Config_to_TOML.restype = None


def convert_config_to_toml(source, destination):
    """Convert an old config (input) file to a toml based config file"""
    _Convert_Config_to_TOML(
        ctypes.c_char_p(source.encode("utf-8")),
        ctypes.c_char_p(destination.encode("utf-8")),
    )
