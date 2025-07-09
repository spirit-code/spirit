"""
Legacy
====================

The legacy module contains functionalty that is provided for backwards compatibility.

This functionalty may be dropped in future releases and should be considered deprecated.
"""

import ctypes
import warnings
import functools

try:
    from warnings import deprecated
except ImportError:

    def deprecated(msg, /, *, category=DeprecationWarning, stacklevel=1):
        """
        Backport of `warnings.deprecated()` decorator introduced in Python 3.13.

        Compared to the builtin version this decorator can onle be used with functions.
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(msg, category=category, stacklevel=stacklevel + 1)
                return func(*args, **kwargs)

            return wrapper

        return decorator


### Load Library
from spirit.spiritlib import _spirit

__all__ = ("deprecated", "convert_config_to_toml")


_Convert_Config_to_TOML = _spirit.Legacy_Convert_Config_to_TOML
_Convert_Config_to_TOML.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_Convert_Config_to_TOML.restype = None


def convert_config_to_toml(source, destination):
    """Convert an old config (input) file to a toml based config file"""
    _Convert_Config_to_TOML(
        ctypes.c_char_p(source.encode("utf-8")),
        ctypes.c_char_p(destination.encode("utf-8")),
    )
