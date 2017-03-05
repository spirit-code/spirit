module spirit

  export State_Setup
  export Simulation_PlayPause

  # TODO: do this as a relative path
  const spiritlib = "C:/Users/Gideon/Git/spirit/core/julia/Spirit/spirit.dll"

  include("state.jl")
  include("simulation.jl")

end
