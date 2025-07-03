push!(LOAD_PATH, "./core/")

using core

cfgfile = ""
# cfgfile = "../input/markus-paper.toml"
# cfgfile = "../input/gideon-master-thesis-isotropic.toml"
# cfgfile = "../input/daniel-master-thesis-isotropic.toml"

p_state = State_Setup(cfgfile)

Simulation_PlayPause(p_state, "LLG", "SIB", 100, 100)
