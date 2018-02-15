import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.LoadSpiritLibrary()


### Send a Log message
_Send          = _spirit.Log_Send
_Send.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, 
                  ctypes.c_int, ctypes.c_int]
_Send.restype  = None
def Send(p_state, level, sender, message, idx_image=-1, idx_chain=-1):
    _Send(ctypes.c_void_p(p_state), ctypes.c_int(level), ctypes.c_int(sender), 
          ctypes.c_char_p(message.encode('utf-8')), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

### Append Log to file
_Append          = _spirit.Log_Append
_Append.argtypes = [ctypes.c_void_p]
_Append.restype  = None
def Append(p_state):
    _Append(ctypes.c_void_p(p_state))

### Get number of Log entries
_Get_N_Entries          = _spirit.Log_Get_N_Entries
_Get_N_Entries.argtypes = [ctypes.c_void_p]
_Get_N_Entries.restype  = None
def GetNEntries(p_state):
    return int(_Get_N_Entries(ctypes.c_void_p(p_state)))

### Get number of error messages
_Get_N_Errors          = _spirit.Log_Get_N_Errors
_Get_N_Errors.argtypes = [ctypes.c_void_p]
_Get_N_Errors.restype  = None
def GetNErrors(p_state):
    return int(_Get_N_Errors(ctypes.c_void_p(p_state)))

### Get number of warning messages
_Get_N_Warnings          = _spirit.Log_Get_N_Warnings
_Get_N_Warnings.argtypes = [ctypes.c_void_p]
_Get_N_Warnings.restype  = None
def GetNWarnings(p_state):
    return int(_Get_N_Warnings(ctypes.c_void_p(p_state)))

### Set the tag in front of the Log file
_Set_Output_File_Tag          = _spirit.Log_Set_Output_File_Tag
_Set_Output_File_Tag.argtypes = [ctypes.c_char_p]
_Set_Output_File_Tag.restype  = None
def SetOutputFileTag(p_state, tag):
    _Set_Output_File_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')))

### Set the tag in front of the Log file
_Set_Output_Folder          = _spirit.Log_Set_Output_Folder
_Set_Output_Folder.argtypes = [ctypes.c_char_p]
_Set_Output_Folder.restype  = None
def SetOutputFolder(p_state, tag):
    _Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')))

### Set whether the Log is output to the console
_Set_Output_To_Console          = _spirit.Log_Set_Output_To_Console
_Set_Output_To_Console.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_Set_Output_To_Console.restype  = None
def SetOutputToConsole(p_state, b):
    _Set_Output_To_Console(ctypes.c_void_p(p_state), ctypes.c_bool(b))

### Set the level up to which the Log is output to the console
_Set_Output_Console_Level          = _spirit.Log_Set_Output_Console_Level
_Set_Output_Console_Level.argtypes = [ctypes.c_void_p, ctypes.c_int]
_Set_Output_Console_Level.restype  = None
def SetOutputConsoleLevel(p_state, level):
    _Set_Output_Console_Level(ctypes.c_void_p(p_state), ctypes.c_int(level))

### Set whether the Log is output to a file
_Set_Output_To_File          = _spirit.Log_Set_Output_To_File
_Set_Output_To_File.argtypes = [ctypes.c_void_p, ctypes.c_bool]
_Set_Output_To_File.restype  = None
def SetOutputToFile(p_state, b):
    _Set_Output_To_File(ctypes.c_void_p(p_state), ctypes.c_bool(b))

### Set the level up to which the Log is output to a file
_Set_Output_File_Level          = _spirit.Log_Set_Output_File_Level
_Set_Output_File_Level.argtypes = [ctypes.c_void_p, ctypes.c_int]
_Set_Output_File_Level.restype  = None
def SetOutputFileLevel(p_state, level):
    _Set_Output_File_Level(ctypes.c_void_p(p_state), ctypes.c_int(level))

### Returns whether the Log is output to the console
_Get_Output_To_Console          = _spirit.Log_Get_Output_To_Console
_Get_Output_To_Console.argtypes = [ctypes.c_void_p]
_Get_Output_To_Console.restype  = ctypes.c_bool
def GetOutputToConsole(p_state):
    return bool(_Get_Output_To_Console(ctypes.c_void_p(p_state)))

### Returns the level up to which the Log is output to the console
_Get_Output_Console_Level          = _spirit.Log_Get_Output_Console_Level
_Get_Output_Console_Level.argtypes = [ctypes.c_void_p]
_Get_Output_Console_Level.restype  = ctypes.c_int
def GetOutputConsoleLevel(p_state):
    return int(_Get_Output_Console_Level(ctypes.c_void_p(p_state)))

### Returns whether the Log is output to a file
_Get_Output_To_File          = _spirit.Log_Get_Output_To_File
_Get_Output_To_File.argtypes = [ctypes.c_void_p]
_Get_Output_To_File.restype  = ctypes.c_bool
def GetOutputToFile(p_state):
    return bool(_Get_Output_To_File(ctypes.c_void_p(p_state)))

### Returns the level up to which the Log is output to a file
_Get_Output_File_Level          = _spirit.Log_Get_Output_File_Level
_Get_Output_File_Level.argtypes = [ctypes.c_void_p]
_Get_Output_File_Level.restype  = ctypes.c_int
def GetOutputFileLevel(p_state):
    return int(_Get_Output_File_Level(ctypes.c_void_p(p_state)))