push!(LOAD_PATH, "C:/Users/Gideon/Git/juliatest/core/")

using core

state = State_Setup()

Simulation_PlayPause(state, "LLG", "SIB", 100, 20)
