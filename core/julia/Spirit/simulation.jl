function Simulation_PlayPause(p_state, method_type, optimizer_type, n_iterations=-1, n_iterations_log=-1, idx_image=-1, idx_chain=-1)
  ccall((:Simulation_PlayPause, spiritlib),  Void, (Ptr{Void}, Cstring, Cstring, Cint, Cint, Cint, Cint), p_state, method_type, optimizer_type, n_iterations, n_iterations_log, idx_image, idx_chain)
end
