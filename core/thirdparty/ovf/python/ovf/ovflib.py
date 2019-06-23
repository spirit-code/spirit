### Get the operating system and according library name
def GetOVFLibName():
    from sys import platform as _platform
    libname = ''
    if _platform == "linux" or _platform == "linux2":
        # Linux
        libname = 'libovf.so'
    elif _platform == "darwin":
        # OS X
        libname = 'libovf.dylib'
    elif _platform == "win32":
        # Windows
        libname = 'ovf.dll'
    return libname

### Get the OVF library as CDLL
def LoadOVFLibrary():
    import os
    import ctypes

    ### Get this file's directory. The library should be here
    ovf_py_dir = os.path.dirname(os.path.realpath(__file__))

    libname = GetOVFLibName()

    ### Load the OVF library
    _ovf = ctypes.CDLL(ovf_py_dir + '/' + libname)

    ### Return
    return _ovf