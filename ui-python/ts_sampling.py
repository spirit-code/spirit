import os
import sys

### Make sure to find the Spirit modules
### This is only needed if you did not install the package
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../core/python"))
sys.path.insert(0, spirit_py_dir)

### Import Spirit modules
from spirit import state, configuration, simulation, io, parameters

cfgfile = "input/input.cfg"
quiet = False

N_SAMPLES       = 2
N_DECORRELATION = 100
N_TOTAL = N_SAMPLES * N_DECORRELATION

temperature = 10 #kb

with state.State(cfgfile, quiet) as p_state:
    io.image_read(p_state, "sp.ovf") # The initial image should be a saddle point, the method will then compute the unstable mode and sample spins configurations which are orthogonal to it after geodesic transport

    parameters.mc.set_temperature(p_state, temperature) # Uses the Monte Carlo parameters

    simulation.start(p_state, simulation.METHOD_TS_SAMPLING, None, single_shot=True, n_iterations=N_TOTAL)

    for i in range(N_SAMPLES):
        simulation.n_shot(p_state, N_DECORRELATION)
        io.image_write(p_state, f"spins_{i}.ovf")