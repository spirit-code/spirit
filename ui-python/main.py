### Make sure to find the interface module
import os
interface_dir = os.path.dirname(os.path.realpath(__file__)) + "/interface"
import sys
sys.path.insert(0, interface_dir)

### Import core library
import interface.core as core


# cfgfile = "input/markus-paper.cfg";
cfgfile = "input/gideon-master-thesis-isotropic.cfg";
# cfgfile = "input/daniel-master-thesis-isotropic.cfg";

### Get a State pointer
state = core.setupState(cfgfile)


### ...
# // Copy the system a few times
# Chain_Image_to_Clipboard(state.get());
# for (int i=1; i<7; ++i)
# {
#     Chain_Insert_Image_After(state.get());
# }


### ...
# // Parameters
# double dir[3] = { 0,0,1 };
# double pos[3] = { 14.5, 14.5, 0 };

# // Read Image from file
# //Configuration_from_File(state.get(), spinsfile, 0);
# // Read Chain from file
# //Chain_from_File(state.get(), chainfile);

# // First image is homogeneous with a Skyrmion at pos
# Configuration_Homogeneous(state.get(), dir, 0);
# Configuration_Skyrmion(state.get(), pos, 6.0, 1.0, -90.0, false, false, false, 0);
# // Last image is homogeneous
# Configuration_Homogeneous(state.get(), dir, state->noi-1);

# // Create transition of images between first and last
# Transition_Homogeneous(state.get(), 0, state->noi-1);


### ...
# Simulation_PlayPause(state)

### ...
# // Finish
# Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
# Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= MonoSpin Finished =================");
# Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
# Log.Append_to_File();