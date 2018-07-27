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
from spirit import hamiltonian
from spirit import quantities
from spirit import geometry
from spirit import constants

import numpy as np
from scipy import special

MC = simulation.METHOD_MC

# Parameters
n_thermalisation = 500
n_decorrelation  = 2 # decorrelation between samples
n_samples        = 25000

n_temperatures   = 60
T_start          = 0.001
T_end            = 15

system_size      = 30

# Expected Tc value (factor 0.5 due to unique pairs instead of neighbours)
order = 3 # Number of unique pairs
Jij = 1   # Exchange per unique pair
Tc = 1.44/2.0 * order * Jij / (3 * constants.k_B) # factor 1.44/2 is expected difference to mean field value
Ta = Tc - 2
Tb = Tc + 2

T_step = (T_end-T_start) / n_temperatures
sample_temperatures = np.linspace(T_start, Ta-0.5*T_step, num=n_temperatures/4)
sample_temperatures = np.append(sample_temperatures, np.linspace( Ta+0.5*T_step, Tb-0.5*T_step, num=n_temperatures - (n_temperatures/4)*2))
sample_temperatures = np.append(sample_temperatures, np.linspace(Tb+0.5*T_step,        T_end,  num=n_temperatures/4))

energy_samples          = []
magnetization_samples   = []
susceptibility_samples  = []
specific_heat_samples   = []
binder_cumulant_samples = []

cfgfile = "ui-python/input.cfg"                   # Input File
with state.State(cfgfile) as p_state: # State setup
    # Set parameters
    hamiltonian.set_field(p_state, 0.0, [0,0,1])
    hamiltonian.set_exchange(p_state, Jij, [1.0])
    hamiltonian.set_dmi(p_state, 0, [])

    parameters.mc.set_output_general(p_state, any=False)           # Disallow any output

    geometry.set_mu_s(p_state, 1.0)
    geometry.set_n_cells(p_state, [system_size, system_size, system_size])

    NOS = system.get_nos(p_state)
    
    # Ferromagnet in z-direction
    configuration.plus_z(p_state)
    # configuration.Random(p_state)

    # Loop over temperatures
    for iT, T in enumerate(sample_temperatures):
        parameters.mc.set_temperature(p_state, T)

        # Cumulative average variables
        E  = 0
        E2 = 0
        M  = 0
        M2 = 0
        M4 = 0

        # Thermalisation
        parameters.mc.set_iterations(p_state, n_thermalisation, n_thermalisation) # We want n_thermalisation iterations and only a single log message
        simulation.start(p_state, MC) # Start a MC simulation

        # Sampling at given temperature
        parameters.mc.set_iterations(p_state, n_decorrelation*n_samples, n_decorrelation*n_samples) # We want n_decorrelation iterations and only a single log message
        simulation.start(p_state, MC, single_shot=True) # Start a single-shot MC simulation
        for n in range(n_samples):
            # Run decorrelation
            for i_decorr in range(n_decorrelation):
                simulation.single_shot(p_state) # one MC iteration
            # Get energy
            E_local = system.get_energy(p_state) / NOS
            # Get magnetization
            M_local = np.array(quantities.get_magnetization(p_state))
            M_local_tot = np.linalg.norm(M_local)
            # Add to cumulative averages
            E   += E_local
            E2  += E_local**2
            M   += M_local_tot
            M2  += M_local_tot**2
            M4  += M_local_tot**4
        # Make sure the MC simulation is not running anymore
        simulation.stop(p_state)

        # Average over samples
        E  /= n_samples
        E2 /= n_samples
        M  /= n_samples
        M2 /= n_samples
        M4 /= n_samples

        # Calculate observables
        chi = (M2 - np.dot(M, M)) / (constants.k_B * T)
        c_v = (E2 - E**2) / (constants.k_B * T**2)
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