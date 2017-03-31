import os
import sys

### Make sure to find the Spirit modules
### This is only needed if you did not install the package
# spirit_py_dir = os.path.dirname(os.path.realpath(__file__)) + "core/python/Spirit"
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../core/python"))
sys.path.insert(0, spirit_py_dir)


### Import numpy
import numpy as np

### Import Spirit modules
from spirit import state
from spirit import system
from spirit import geometry
from spirit import chain
from spirit import configuration
from spirit import transition
from spirit import simulation
from spirit import quantities
from spirit import io
from spirit import log


def configurations(p_state):
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
    configuration.PlusZ(p_state, idx_image=0)
    configuration.Skyrmion(p_state, 5.0, phase=-90.0, idx_image=0)
    ### Last image is homogeneous
    configuration.PlusZ(p_state, idx_image=noi-1)

    # spinsfile = "input/spins.txt"
    # io.Image_Read(p_state, spinsfile)

    ### Create transition of images between first and last
    transition.Homogeneous(p_state, 0, noi-1)



def simulations(p_state):
    simulation.PlayPause(p_state, "LLG", "SIB")



def evaluations(p_state):
    ### Order Parameter: total magnetization
    M = quantities.Get_Magnetization(p_state)
    # TODO: save order parameter

    ### Total Energy
    # TODO
    E = system.Get_Energy(p_state)



### Save snapshot of a system slice
def Run_PovRay_Script(scriptname, w=1000, h=1000):
    from subprocess import call
    # create tmp folder
    command = "mkdir -p /tmp/PovRay/"
    call(command, shell=True)
    # create the image
    command = "povray "+scriptname+".pov -w"+str(w)+" -h"+str(h)+" +aa0.3 -O/tmp/PovRay/"+scriptname+".png"
    call(command, shell=True)
    # copy to folder
    command = "cp /tmp/PovRay/"+scriptname+".png ."
    call(command, shell=True)

def snapshots(p_state):
    ### Test
    na, nb, nc = geometry.Get_N_Cells(p_state)
    spins = system.Get_Spin_Directions(p_state)
    spins.shape = (nc, nb, na, 3)
    positions = geometry.Get_Spin_Positions(p_state)
    positions.shape = (nc, nb, na, 3)

    n_layer_spins = nb*nc
    n_layers = na
    if(n_layers > 2):
        layer_x = n_layers/2
    else:
        layer_x = 1

    n_layer_spins = na*nc
    n_layers = nb
    if(n_layers > 2):
        layer_y = n_layers/2
    else:
        layer_y = 1

    n_layer_spins = na*nb
    n_layers = nc
    if(n_layers > 2):
        layer_z = n_layers/2
    else:
        layer_z = 1

    print(layer_x, layer_y, layer_z)

    with open("data_x.txt", "w") as outfile:
        for j in range(nc):
            for i in range(nb):
                outfile.write("%s, %s, %s, %s, %s, %s,\n" % (positions[j,i,layer_x,1], positions[j,i,layer_x,2], 0.0, spins[j,i,layer_x,1], spins[j,i,layer_x,2], spins[j,i,layer_x,0]))

    with open("data_y.txt", "w") as outfile:
        for j in range(nc):
            for i in range(na):
                outfile.write("%s, %s, %s, %s, %s, %s,\n" % (positions[j,layer_y,i,0], positions[j,layer_y,i,2], 0.0, spins[j,layer_y,i,0], spins[j,layer_y,i,2], spins[j,layer_y,i,1]))

    with open("data_z.txt", "w") as outfile:
        for j in range(nb):
            for i in range(na):
                outfile.write("%s, %s, %s, %s, %s, %s,\n" % (positions[layer_z,j,i,0], positions[layer_z,j,i,1], 0.0, spins[layer_z,j,i,0], spins[layer_z,j,i,1], spins[layer_z,j,i,2]))
    
    from subprocess import call
    call("cp data_x.txt data.txt", shell=True)
    Run_PovRay_Script("Show_Spins", 1000, 1000)
    call("cp Show_Spins.png spins_x.png", shell=True)

    call("cp data_y.txt data.txt", shell=True)
    Run_PovRay_Script("Show_Spins", 1000, 1000)
    call("cp Show_Spins.png spins_y.png", shell=True)

    call("cp data_z.txt data.txt", shell=True)
    Run_PovRay_Script("Show_Spins", 1000, 1000)
    call("cp Show_Spins.png spins_z.png", shell=True)

    call("rm -f Data.txt Show_Spins.png", shell=True)



cfgfile = "input/input.cfg"
# cfgfile = "input/markus-paper.cfg"
# cfgfile = "input/gideon-master-thesis-anisotropic.cfg"
# cfgfile = "input/daniel-master-thesis-isotropic.cfg"

with state.State(cfgfile) as p_state:
    ### Setup initial configurations
    configurations(p_state)
    ### Run simulations
    simulations(p_state)
    ### Evaluate the results
    # evaluations(p_state)
    ### Create snapshots of the system
    # snapshots(p_state)