__all__ = ("_spirit", "wrap_function")


### Get the operating system and according library name
def _get_spirit_lib_name():
    from sys import platform as _platform

    libname = ""
    if _platform == "linux" or _platform == "linux2":
        # Linux
        libname = "libSpirit.so"
    elif _platform == "darwin":
        # OS X
        libname = "libSpirit.dylib"
    elif _platform == "win32":
        # Windows
        libname = "Spirit.dll"
    return libname


### Get the Spirit library as CDLL
def _load_spirit_library():
    import os
    import ctypes

    ### Get this file's directory. The library should be here
    spirit_py_dir = os.path.dirname(os.path.realpath(__file__))

    libname = _get_spirit_lib_name()

    ### Load the Spirit library
    return ctypes.CDLL(os.path.join(spirit_py_dir, libname))


### store library in a module variable such that the import system may load it only once
_spirit = _load_spirit_library()


### Wrap a function in a thread for it to be interruptible
def wrap_function(function, arguments):
    import threading

    t = threading.Thread(target=function, args=arguments)
    t.daemon = True
    t.start()
    while t.is_alive():  # wait for the thread to exit
        t.join(0.1)
