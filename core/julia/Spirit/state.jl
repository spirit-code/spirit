function State_Setup(config_file="")
  return ccall((:State_Setup, spiritlib),  Ptr{Void}, (Cstring,), config_file)
end
