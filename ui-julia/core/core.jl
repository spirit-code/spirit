module core

  export State_Setup
  export Simulation_PlayPause

  # TODO: do this as a relative path
  const corelib = "C:/Users/Gideon/Git/monospin/ui-julia/core/core.dll"

  include("state.jl")
  include("simulation.jl")

end
