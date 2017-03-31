import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### TODO:
#Hamiltonian_Set_mu_s
#Hamiltonian_Set_DMI
#Hamiltonian_Set_BQE
#Hamiltonian_Set_FSC
#all getters (including name!)

### Set external magnetic field
_Set_Field             = _spirit.Hamiltonian_Set_Field
_Set_Field.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Set_Field.restype     = None
def Set_Field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Set_Field(p_state, ctypes.c_float(magnitude), vec3(*direction), idx_image, idx_chain)

### Set anisotropy
_Set_Anisotropy             = _spirit.Hamiltonian_Set_Anisotropy
_Set_Anisotropy.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Set_Anisotropy.restype     = None
def Set_Anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Set_Anisotropy(p_state, ctypes.c_float(magnitude), vec3(*direction), idx_image, idx_chain)

### Set spin transfer torque
_Set_STT             = _spirit.Hamiltonian_Set_STT
_Set_STT.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Set_STT.restype     = None
def Set_STT(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Set_STT(p_state, ctypes.c_float(magnitude), vec3(*direction), idx_image, idx_chain)

### Set Temperature
_Set_Temperature             = _spirit.Hamiltonian_Set_Temperature
_Set_Temperature.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
_Set_Temperature.restype     = None
def Set_Temperature(p_state, temperature, idx_image=-1, idx_chain=-1):
    _Set_Temperature(p_state, ctypes.c_float(temperature), idx_image, idx_chain)