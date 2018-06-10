import sys
import os
from datetime import datetime
 
### Make sure to find the Spirit modules
### This is only needed if you did not install the package
spirit_py_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), "../core/python"))
sys.path.insert(0, spirit_py_dir)
 
from spirit import state
from spirit import system
from spirit import simulation
from spirit import configuration
from spirit import parameters
from spirit import quantities
from spirit import geometry
from spirit import constants

import numpy as np
from scipy import special

# Parameters
n_thermalisation = 500
n_steps          = 2 # decorrelation between samples
n_samples        = 25000

n_temperatures   = 60
T_start          = 0.001
T_end            = 15

system_size      = 30

# Expected Tc value (factor 0.5 due to unique pairs instead of neighbours)
order = 3 # Number of unique pairs
Jij = 1   # Exchange per unique pair
Tc = 0.5 * order * Jij / (3 * constants.k_B())

T_step = (T_end-T_start) / n_temperatures
sample_temperatures = np.linspace(T_start, 6-0.5*T_step, num=n_temperatures/4)
sample_temperatures = np.append(sample_temperatures, np.linspace( 6+0.5*T_step, 10-0.5*T_step, num=n_temperatures - (n_temperatures/4)*2))
sample_temperatures = np.append(sample_temperatures, np.linspace(10+0.5*T_step,        T_end,  num=n_temperatures/4))

energy_samples          = []
magnetization_samples   = []
susceptibility_samples  = []
specific_heat_samples   = []
binder_cumulant_samples = []

cfgfile = "ui-python/input.cfg"                   # Input File
with state.State(cfgfile, quiet=True) as p_state: # State setup
    # Set parameters
    parameters.mc.setOutputGeneral(p_state, False)         # Disallow any output
    geometry.setNCells(p_state, [system_size, system_size, system_size])

    NOS = system.Get_NOS(p_state)
    
    # Ferromagnet in z-direction
    configuration.PlusZ(p_state)
    # configuration.Random(p_state)

    # Loop over temperatures
    for iT, T in enumerate(sample_temperatures):
        parameters.mc.setTemperature(p_state, T)

        # Cumulative average variables
        E  = 0
        E2 = 0
        M  = 0
        M2 = 0
        M4 = 0

        # Thermalisation
        parameters.mc.setIterations(p_state, n_thermalisation, n_thermalisation) # We want n_steps iterations and only a single log message
        simulation.PlayPause(p_state, "MC", "") # Start a MC simulation
        parameters.mc.setIterations(p_state, n_steps, n_steps) # We want n_steps iterations and only a single log message
        # Sampling at given temperature
        for n in range(n_samples):
            # Run decorrelation
            simulation.PlayPause(p_state, "MC", "") # Start a MC simulation
            # Get energy
            E_local = system.Get_Energy(p_state)/NOS
            # Get magnetization
            M_local = np.array(quantities.Get_Magnetization(p_state))
            M_local_tot = np.linalg.norm(M_local)
            # Add to cumulative averages
            E   += E_local
            E2  += E_local**2
            M   += M_local_tot
            M2  += M_local_tot**2
            M4  += M_local_tot**4

        # Average over samples
        E  /= n_samples
        E2 /= n_samples
        M  /= n_samples
        M2 /= n_samples
        M4 /= n_samples

        # Calculate observables
        chi = (M2 - np.dot(M, M)) / (constants.k_B() * T)
        c_v = (E2 - E**2) / (constants.k_B() * T**2)
        cumulant = 1 - M4/(3 * M2**2)

        # Output
        strprint = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + " " + str(iT) + "/" + str(n_temperatures) + " T=" + str(T) + " E=" + str(E) + " M=" + str(M) + " chi=" + str(chi) + " c_v=" + str(c_v) + " cumulant=" + str(cumulant)
        print(strprint)

        energy_samples.append(E)
        magnetization_samples.append(M)
        susceptibility_samples.append(chi)
        specific_heat_samples.append(c_v)
        binder_cumulant_samples.append(cumulant)

# Print out samples
for i, T in enumerate(sample_temperatures):
    strprint = "T=" + str(T) + ": E=" + str(energy_samples[i]) + " M=" + str(magnetization_samples[i]) + " chi=" + str(susceptibility_samples[i]) + " c_v=" + str(specific_heat_samples[i]) + " cumulant=" + str(binder_cumulant_samples[i])
    print(strprint)

print("Expected critical temperature: " + str(Tc))

# Write output file
with open("output/output_mc.txt", "w") as f:
    f.write( str(0) + "     " + str(1) )
    for i, T in enumerate(sample_temperatures):
        f.write( "\n" + str(T) + "     " + str(energy_samples[i]) + "     " + str(magnetization_samples[i]) + "     " + str(susceptibility_samples[i]) + "     " + str(specific_heat_samples[i]) + "     " + str(binder_cumulant_samples[i]) )