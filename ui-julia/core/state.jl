function State_Setup(config_file="")
  state = ccall((:State_Setup, corelib),  Ptr{Void}, (Cstring,), config_file)
end
