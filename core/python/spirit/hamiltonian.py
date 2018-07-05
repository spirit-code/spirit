import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()

### ---------------------------------- Set ----------------------------------

### Set boundary conditions
_Set_Boundary_Conditions             = _spirit.Hamiltonian_Set_Boundary_Conditions
_Set_Boundary_Conditions.argtypes    = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool),
                          ctypes.c_int, ctypes.c_int]
_Set_Boundary_Conditions.restype     = None
def Set_Boundary_Conditions(p_state, boundaries, idx_image=-1, idx_chain=-1):
    bool3 = ctypes.c_bool * 3
    _Set_Boundary_Conditions(ctypes.c_void_p(p_state), bool3(*boundaries),
                             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set magnetic moment globally
_Set_mu_s             = _spirit.Hamiltonian_Set_mu_s
_Set_mu_s.argtypes    = [ctypes.c_void_p, ctypes.c_float,
                          ctypes.c_int, ctypes.c_int]
_Set_mu_s.restype     = None
def Set_mu_s(p_state, mu_s, idx_image=-1, idx_chain=-1):
    _Set_mu_s(ctypes.c_void_p(p_state), ctypes.c_float(mu_s),
              ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set global external magnetic field
_Set_Field             = _spirit.Hamiltonian_Set_Field
_Set_Field.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int, ctypes.c_int]
_Set_Field.restype     = None
def Set_Field(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Set_Field(ctypes.c_void_p(p_state), ctypes.c_float(magnitude), vec3(*direction),
               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set global anisotropy
_Set_Anisotropy             = _spirit.Hamiltonian_Set_Anisotropy
_Set_Anisotropy.argtypes    = [ctypes.c_void_p, ctypes.c_float, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Set_Anisotropy.restype     = None
def Set_Anisotropy(p_state, magnitude, direction, idx_image=-1, idx_chain=-1):
    vec3 = ctypes.c_float * 3
    _Set_Anisotropy(ctypes.c_void_p(p_state), ctypes.c_float(magnitude), vec3(*direction), 
                    ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set exchange interaction in form of neighbour shells
_Set_Exchange             = _spirit.Hamiltonian_Set_Exchange
_Set_Exchange.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                               ctypes.c_int, ctypes.c_int]
_Set_Exchange.restype     = None
def Set_Exchange(p_state, n_shells, J_ij, idx_image=-1, idx_chain=-1):
    vec = ctypes.c_float * n_shells
    _Set_Exchange(ctypes.c_void_p(p_state), ctypes.c_int(n_shells), vec(*J_ij),
                  ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set Dzyaloshinskii-Moriya interaction in form of neighbour shells
_Set_DMI             = _spirit.Hamiltonian_Set_DMI
_Set_DMI.argtypes    = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int, ctypes.c_int, ctypes.c_int]
_Set_DMI.restype     = None
def Set_DMI(p_state, n_shells, D_ij, chirality=1, idx_image=-1, idx_chain=-1):
    vec = ctypes.c_float * n_shells
    _Set_DMI(ctypes.c_void_p(p_state), ctypes.c_int(n_shells), vec(*D_ij),
             ctypes.c_int(chirality), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Set dipole-dipole interaction in form of exact calculation within a cutoff radius
_Set_DDI             = _spirit.Hamiltonian_Set_DDI
_Set_DDI.argtypes    = [ctypes.c_void_p, ctypes.c_float,
                        ctypes.c_int, ctypes.c_int]
_Set_DDI.restype     = None
def Set_DDI(p_state, radius, idx_image=-1, idx_chain=-1):
    _Set_DDI(ctypes.c_void_p(p_state), ctypes.c_float(radius),
             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### ---------------------------------- Get ----------------------------------

### Get the name of the Hamiltonian
_Get_Name          = _spirit.Hamiltonian_Get_Name
_Get_Name.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_Name.restype  = ctypes.c_char_p
def Get_Name(p_state, idx_image=-1, idx_chain=-1):
    return str(_Get_Name(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

### Get the boundary conditions [a, b, c]
_Get_Boundary_Conditions          = _spirit.Hamiltonian_Get_Boundary_Conditions
_Get_Boundary_Conditions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_bool),
                                     ctypes.c_int, ctypes.c_int]
_Get_Boundary_Conditions.restype  = None
def Get_Boundary_Conditions(p_state, idx_image=-1, idx_chain=-1):
    boundaries = (3*ctypes.c_bool)()
    _Get_Boundary_Conditions(ctypes.c_void_p(p_state), boundaries,
                             ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return [bc for bc in boundaries]

### Get the global external magnetic field
_Get_Field          = _spirit.Hamiltonian_Get_Field
_Get_Field.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                       ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int]
_Get_Field.restype  = None
def Get_Field(p_state, idx_image=-1, idx_chain=-1):
    magnitude = (1*ctypes.c_float)()
    normal = (3*ctypes.c_float)()
    _Get_Field(ctypes.c_void_p(p_state), magnitude, normal,
               ctypes.c_int(idx_image), ctypes.c_int(idx_chain))
    return float(magnitude), [n for n in normal]

### Get dipole-dipole interaction cutoff radius
_Get_DDI          = _spirit.Hamiltonian_Get_DDI
_Get_DDI.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_Get_DDI.restype  = ctypes.c_float
def Get_DDI(p_state, idx_image=-1, idx_chain=-1):
    return float(_Get_DDI(ctypes.c_void_p(p_state), ctypes.c_int(idx_image), 
                          ctypes.c_int(idx_chain)))