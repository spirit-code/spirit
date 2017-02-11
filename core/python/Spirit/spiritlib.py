def LoadSpiritLibrary():
    import os
    import ctypes
    from sys import platform as _platform

    ### Get this file's directory. The library should be here
    spirit_py_dir = os.path.dirname(os.path.realpath(__file__))

    libname = ''
    ### Get the Operating System and set lib name accordingly
    if _platform == "linux" or _platform == "linux2":
        # Linux
        libname = 'libSpirit.so'
    elif _platform == "darwin":
        # OS X
        libname = 'libSpirit.dylib'
    elif _platform == "win32":
        # Windows
        libname = 'Spirit.dll'

    ### Load the Spirit library
    _spirit = ctypes.CDLL(spirit_py_dir + '/' + libname)

    ### Return
    return _spirit

### Wrap a function in a thread for it to be interruptible
def WrapFunction(function, arguments):
    import threading
    t = threading.Thread(target=function, args=arguments)
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exit
        t.join(.1)