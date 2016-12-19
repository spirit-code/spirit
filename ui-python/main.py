### Make sure to find the core modules
import os
core_dir = os.path.dirname(os.path.realpath(__file__)) + "/ui-python"
# core_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "..", "/ui-python/core"))
import sys
sys.path.insert(0, core_dir)

### Import numpy
import numpy as np

### Import core library
from core import state
from core import system
from core import geometry
from core import chain
from core import configuration
from core import transition
from core import simulation
from core import quantities
from core import io
from core import log


# cfgfile = "input/markus-paper.cfg"
cfgfile = "input/gideon-master-thesis-anisotropic.cfg"
# cfgfile = "input/daniel-master-thesis-isotropic.cfg"

### Get a State pointer
p_state = state.setup(cfgfile)

### Copy the system a few times
chain.Image_to_Clipboard(p_state)
for number in range(1,7):
    chain.Insert_Image_After(p_state)
noi = chain.Get_NOI(p_state)

### Read Image from file
# Configuration_from_File(state.get(), spinsfile, 0);
### Read Chain from file
# Chain_from_File(state.get(), chainfile);

### First image is homogeneous with a Skyrmion at pos
configuration.PlusZ(p_state, 0)
configuration.Skyrmion(p_state, [0,0,0], 5.0, 1, -90.0, False, False, False, 0)
### Last image is homogeneous
configuration.PlusZ(p_state, noi-1)

# spinsfile = "input/spins.txt"
# io.Image_Read(p_state, spinsfile)

### Create transition of images between first and last
transition.Homogeneous(p_state, 0, noi-1)

### Run a LLG simulation
###     We use a thread, so that KeyboardInterrupt can be forwarded to the CDLL call
###     We might want to think about using PyDLL and about a signal handler in the core library
###     see here: http://stackoverflow.com/questions/14271697/ctrlc-doesnt-interrupt-call-to-shared-library-using-ctypes-in-python
simulation.PlayPause(p_state, "LLG", "SIB")

# ### Save some data
# nos = system.Get_NOS(p_state)
# spins = system.Get_Spin_Directions(p_state)
# positions = geometry.Get_Spin_Positions(p_state)

# # get the n'th layer in z-direction
# na, nb, nc = geometry.Get_N_Cells(p_state)
# n_layer_spins = na*nb
# n_layers = nc
# if(n_layers > 2):
#     layer = n_layers/2
# else:
#     layer = 1
# print(n_layers, " ", layer)

# slice_spins = spins[n_layer_spins*(layer-1):n_layer_spins*layer]
# slice_positions = positions[n_layer_spins*(layer-1):n_layer_spins*layer]
# print(slice_spins)


# thefile = open("test.txt", "w")
# for i in range(n_layer_spins):
#     thefile.write("%s, %s, %s, %s, %s, %s,\n" % (slice_positions[i,0], slice_positions[i,1], 0, slice_spins[i,0], slice_spins[i,1], slice_spins[i,2]))

# ### Order Parameter: total magnetization
# M = quantities.Get_Magnetization(p_state)


### Finish
log.Send(p_state, 0, 0, "=====================================================")
log.Send(p_state, 0, 0, "================= Spirit Finished ===================")
log.Send(p_state, 0, 0, "=====================================================")
log.Append(p_state)