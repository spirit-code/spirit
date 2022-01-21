import os
import sys

### Make sure to find the Spirit modules
### This is only needed if you did not install the package
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "core/python"))
print(spirit_py_dir)
sys.path.insert(0, spirit_py_dir)

### Import Spirit modules
from spirit import state
from spirit import configuration
from spirit import simulation
from spirit import version
from spirit import hamiltonian
from spirit import geometry

print(version.cuda)

N_CELLS  = [6, 6, 1]
N_SHELLS = 2
NOS      = N_CELLS[0] * N_CELLS[1] * N_CELLS[2]

with state.State("", False) as p_state:
    n_cells = geometry.set_n_cells(p_state, N_CELLS)
    hamiltonian.set_exchange(p_state, N_SHELLS, [1, 1]) # Nearest neighbours 
    simulation.start(p_state, simulation.METHOD_MC, 0, n_iterations=1)

import numpy as np

data = np.loadtxt("mc_access_order.txt")
data = data[ data[:,-1].argsort() ][:64] # sort by order of access
import matplotlib.pyplot as plt
# Visualise the order of access
plt.plot( data[:,0], data[:,1] )
plt.xlabel("a")
plt.xlabel("b")
plt.savefig("mc_access_order.png", dpi=300)