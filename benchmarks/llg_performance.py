#!/bin/python

import os
import sys

### Make sure to find the Spirit modules
### This is only needed if you did not install the package
# spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/python"))
# sys.path.insert(0, spirit_py_dir)


from spirit import state
from spirit import geometry
from spirit import configuration
from spirit import simulation
from spirit import parameters
from spirit import hamiltonian
from spirit import io


CFG_FILE = "input/input.cfg"
QUIET = False
NITERS = 1000


with state.State(CFG_FILE) as p_state:
    # spinsfile = "input/spins.ovf"
    # io.image_from_file(state.get(), spinsfile, idx_image=0);

    geometry.set_n_cells(p_state, n_cells=[100,100,20])

    configuration.plus_z(p_state, idx_image=0)
    configuration.hopfion(p_state, 40.0)
    # configuration.skyrmion(p_state, 40.0, phase=-90.0, idx_image=0)

    hamiltonian.set_field(p_state, 5.0, [0,0,1])

    # parameters.llg.set_direct_minimization(p_state, True)
    parameters.llg.set_iterations(p_state, NITERS, NITERS//4)

    simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_DEPONDT)
    # simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP)

    # spirit.io.FILEFORMAT_OVF_TEXT= 3



# GNEB method
# This method operates on a transition between two spin configurations,
# discretised by “images” on a “chain”.
# The procedure follows these steps:
#  - set the number of images
#  - set the initial and final spin configuration
#  - create an initial guess for the transition path
#  - run an initial GNEB relaxation
#  - determine and set the suitable images on the chain to converge on extrema
#  - run a full GNEB relaxation using climbing and falling images
#  - from spirit import state, chain, configuration, transition, simulation

# noi = 7
#
# with state.State(CFG_FILE) as p_state:
#     ### Copy the first image and set chain length
#     chain.image_to_clipboard(p_state)
#     chain.set_length(p_state, noi)
#
#     ### First image is homogeneous with a Skyrmion in the center
#     configuration.plus_z(p_state, idx_image=0)
#     configuration.skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
#     simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=0)
#     ### Last image is homogeneous
#     configuration.plus_z(p_state, idx_image=noi-1)
#     simulation.start(p_state, simulation.METHOD_LLG, simulation.SOLVER_VP, idx_image=noi-1)
#
#     ### Create initial guess for transition: homogeneous rotation
#     transition.homogeneous(p_state, 0, noi-1)
#
#     ### Initial GNEB relaxation
#     simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP, n_iterations=5000)
#     ### Automatically set climbing and falling images
#     chain.set_image_type_automatically(p_state)
#     ### Full GNEB relaxation
#     simulation.start(p_state, simulation.METHOD_GNEB, simulation.SOLVER_VP)
