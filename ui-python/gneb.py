import os
import sys

### Make sure to find the Spirit modules
### This is only needed if you did not install the package
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../core/python"))
sys.path.insert(0, spirit_py_dir)

### Import Spirit modules
from spirit import state
from spirit import system
from spirit import geometry
from spirit import chain
from spirit import configuration
from spirit import transition
from spirit import simulation
from spirit import parameters
from spirit import io
from spirit import log

cfgfile = "input/input.cfg"
quiet = False

with state.State(cfgfile, quiet) as p_state:
    noi = 7
    log.send(p_state, log.LEVEL_ALL, log.SENDER_ALL, "Performing skyrmion collapse calculation with {} images".format(noi))

    ### Set the length of the chain
    chain.image_to_clipboard(p_state)
    chain.set_length(p_state, noi)

    ### Read Image from file
    # spinsfile = "input/spins.ovf"
    # io.image_from_file(state.get(), spinsfile, idx_image=0);
    ### Read Chain from file
    # io.chain_from_file(state.get(), chainfile);

    ### First image is homogeneous with a skyrmion in the center
    configuration.plus_z(p_state, idx_image=0)
    configuration.skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
    ### Last image is homogeneous
    configuration.plus_z(p_state, idx_image=noi-1)

    ### Initial guess: homogeneous transition between first and last image
    transition.homogeneous(p_state, 0, noi-1)

    ### Energy minimisation of first and last image
    LLG = simulation.METHOD_LLG
    GNEB = simulation.METHOD_GNEB
    VP = simulation.SOLVER_VP # Velocity projection minimiser
    simulation.start(p_state, LLG, VP, idx_image=0)
    simulation.start(p_state, LLG, VP, idx_image=noi-1)

    ### Initial relaxation of transition path
    simulation.start(p_state, GNEB, VP, n_iterations=10000)
    ### Full relaxation with climbing image
    parameters.gneb.set_image_type_automatically(p_state)
    simulation.start(p_state, GNEB, VP)

    ### Calculate the energy barrier of the transition
    E = chain.get_energy(p_state)
    delta = max(E) - E[0]
    log.send(p_state, log.LEVEL_ALL, log.SENDER_ALL, "Energy barrier: {} meV".format(delta))