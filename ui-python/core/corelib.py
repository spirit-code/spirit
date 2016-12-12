def LoadCoreLibrary():
    import os
    import ctypes
    from sys import platform as _platform

    ### Get this file's directory. The library should be here
    core_dir = os.path.dirname(os.path.realpath(__file__))

    libname = ''
    ### Get the Operating System and set lib name accordingly
    if _platform == "linux" or _platform == "linux2":
        # Linux
        libname = 'libcore.so'
    elif _platform == "darwin":
        # OS X
        libname = 'libcore.dylib'
    elif _platform == "win32":
        # Windows
        libname = 'core.dll'

    ### Load the core library
    _core = ctypes.CDLL(core_dir + '/' + libname)

    ### Return
    return _core

### Wrap a function in a thread for it to be interruptible
def WrapFunction(function, arguments):
    import threading
    t = threading.Thread(target=function, args=arguments)
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exit
        t.join(.1)