### Make sure to find the core modules
import os
core_dir = os.path.dirname(os.path.realpath(__file__)) + '/core'
import sys
sys.path.insert(0, core_dir)


### Import core library
from core import state
from core import chain
from core import configuration
from core import simulation


# cfgfile = b'input/markus-paper.cfg'
cfgfile = b'input/gideon-master-thesis-isotropic.cfg'
# cfgfile = b'input/daniel-master-thesis-isotropic.cfg'

### Get a State pointer
p_state = state.setup(cfgfile)

if (simulation.Running_LLG(p_state)):
    print "running"
else:
    print "not running"

### Copy the system a few times
chain.Image_to_Clipboard(p_state)
for number in range(1,7):
    chain.Insert_Image_After(p_state)

### Read Image from file
# Configuration_from_File(state.get(), spinsfile, 0);
### Read Chain from file
# Chain_from_File(state.get(), chainfile);

### First image is homogeneous with a Skyrmion at pos
configuration.PlusZ(p_state, 0)
configuration.Skyrmion(p_state, [14.5, 14.5, 0], 5.0, 1, -90.0, False, False, False, 0)
### Last image is homogeneous
configuration.PlusZ(p_state, 6)

### Create transition of images between first and last
# Transition_Homogeneous(state.get(), 0, state->noi-1);


### Run a LLG simulation
# simulation.PlayPause(p_state, "LLG", "SIB")

### ...
# // Finish
# Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
# Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= MonoSpin Finished =================");
# Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
# Log.Append_to_File();