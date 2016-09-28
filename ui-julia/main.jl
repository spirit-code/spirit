push!(LOAD_PATH, "./core/")

using core

cfgfile = ""
# cfgfile = "../input/markus-paper.cfg"
# cfgfile = "../input/gideon-master-thesis-isotropic.cfg"
# cfgfile = "../input/daniel-master-thesis-isotropic.cfg"

p_state = State_Setup(cfgfile)

Simulation_PlayPause(p_state, "LLG", "SIB", 100, 100)
