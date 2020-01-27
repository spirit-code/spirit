import os
import sys

### Make sure to find the Spirit modules
### This is only needed if you did not install the package
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../core/python"))
sys.path.insert(0, spirit_py_dir)

### Import Spirit modules
from spirit import state
from spirit import configuration
from spirit import simulation
from spirit import hamiltonian
from spirit import io

cfgfile = "../input/input.cfg"
quiet = False

with state.State(cfgfile, quiet) as p_state:
    ### Read Image from file
    # spinsfile = "input/spins.ovf"
    # io.image_from_file(state.get(), spinsfile, idx_image=0);

    ### First image is homogeneous with a skyrmion in the center
    configuration.plus_z(p_state, idx_image=0)
    configuration.skyrmion(p_state, 200.0, phase=0.0, idx_image=0)
    configuration.set_region(p_state, region_id=1, pos=[0,0,0], border_rectangular=[100,100,1],idx_image=0)
    hamiltonian.set_field_m(p_state, 200, [0,0,1], 0) 
    hamiltonian.set_field_m(p_state, 200, [0,0,-1], 1) 
    ### LLG dynamics simulation
    LLG = simulation.METHOD_LLG
    LBFGS_OSO = simulation.SOLVER_LBFGS_OSO # Velocity projection minimiser
    
    simulation.start(p_state, LLG, LBFGS_OSO)
    io.image_write(p_state, "out.ovf")
