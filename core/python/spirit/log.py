"""
Log
====================
"""

import spirit.spiritlib as spiritlib
import ctypes

### Load Library
_spirit = spiritlib.load_spirit_library()

# Log levels
LEVEL_ALL       = 0
LEVEL_SEVERE    = 1
LEVEL_ERROR     = 2
LEVEL_WARNING   = 3
LEVEL_PARAMETER = 4
LEVEL_INFO      = 5
LEVEL_DEBUG     = 6

# Log message senders
SENDER_ALL  = 0
SENDER_IO   = 1
SENDER_GNEB = 2
SENDER_LLG  = 3
SENDER_MC   = 4
SENDER_MMF  = 5
SENDER_API  = 6
SENDER_UI   = 7

_Send          = _spirit.Log_Send
_Send.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                  ctypes.c_int, ctypes.c_int]
_Send.restype  = None
def send(p_state, level, sender, message, idx_image=-1, idx_chain=-1):
    """Add a message to the log.

    - level: see integers defined above. The message may be printed to the console and/or written
      to the log file, depending on the current log parameters
    - sender: see integers defined above. Used to distinguish context
    - message: a string which to log
    - idx_image: can be used to specify to which image the message relates (default: active image)
    """
    _Send(ctypes.c_void_p(p_state), ctypes.c_int(level), ctypes.c_int(sender),
          ctypes.c_char_p(message.encode('utf-8')), ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_Append          = _spirit.Log_Append
_Append.argtypes = [ctypes.c_void_p]
_Append.restype  = None
def append(p_state):
    """Force the appending of new messages to the log file."""
    _Append(ctypes.c_void_p(p_state))

_Get_N_Entries          = _spirit.Log_Get_N_Entries
_Get_N_Entries.argtypes = [ctypes.c_void_p]
_Get_N_Entries.restype  = ctypes.c_int
def get_n_entries(p_state):
    """Returns the number of Log entries."""
    return int(_Get_N_Entries(ctypes.c_void_p(p_state)))

_Get_N_Errors          = _spirit.Log_Get_N_Errors
_Get_N_Errors.argtypes = [ctypes.c_void_p]
_Get_N_Errors.restype  = ctypes.c_int
def get_n_errors(p_state):
    """Returns the number of error messages that have been logged."""
    return int(_Get_N_Errors(ctypes.c_void_p(p_state)))

_Get_N_Warnings          = _spirit.Log_Get_N_Warnings
_Get_N_Warnings.argtypes = [ctypes.c_void_p]
_Get_N_Warnings.restype  = ctypes.c_int
def get_n_warnings(p_state):
    """Returns the number of warning messages that have been logged."""
    return int(_Get_N_Warnings(ctypes.c_void_p(p_state)))

_Set_Output_File_Tag          = _spirit.Log_Set_Output_File_Tag
_Set_Output_File_Tag.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_Set_Output_File_Tag.restype  = None
def set_output_file_tag(p_state, tag):
    """Set the tagging string which is placed in front of the log file.

    If "<time>" is used for the tag, it will be the time of creation of the state.
    """
    _Set_Output_File_Tag(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')))

_Set_Output_Folder          = _spirit.Log_Set_Output_Folder
_Set_Output_Folder.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_Set_Output_Folder.restype  = None
def set_output_folder(p_state, tag):
    """Set the output folder in which to place the log file."""
    _Set_Output_Folder(ctypes.c_void_p(p_state), ctypes.c_char_p(tag.encode('utf-8')))

_Set_Output_To_Console          = _spirit.Log_Set_Output_To_Console
_Set_Output_To_Console.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int]
_Set_Output_To_Console.restype  = None
def set_output_to_console(p_state, output, level):
    """Set whether the Log is output to the console and the level up to which messages are logged."""
    _Set_Output_To_Console(ctypes.c_void_p(p_state), ctypes.c_bool(output), ctypes.c_int(level))

_Set_Output_To_File          = _spirit.Log_Set_Output_To_File
_Set_Output_To_File.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int]
_Set_Output_To_File.restype  = None
def set_output_to_file(p_state, output, level):
    """Set whether the Log is output to a file and the level up to which messages are logged."""
    _Set_Output_To_File(ctypes.c_void_p(p_state), ctypes.c_bool(output), ctypes.c_int(level))

_Get_Output_To_Console          = _spirit.Log_Get_Output_To_Console
_Get_Output_To_Console.argtypes = [ctypes.c_void_p]
_Get_Output_To_Console.restype  = ctypes.c_bool
def get_output_to_console(p_state):
    """Returns a bool indicating whether the Log is output to the console."""
    return bool(_Get_Output_To_Console(ctypes.c_void_p(p_state)))

_Get_Output_Console_Level          = _spirit.Log_Get_Output_Console_Level
_Get_Output_Console_Level.argtypes = [ctypes.c_void_p]
_Get_Output_Console_Level.restype  = ctypes.c_int
def get_output_console_level(p_state):
    """Returns the level up to which the Log is output to the console.

    The return value will be one of the integers defined above.
    """
    return int(_Get_Output_Console_Level(ctypes.c_void_p(p_state)))

_Get_Output_To_File          = _spirit.Log_Get_Output_To_File
_Get_Output_To_File.argtypes = [ctypes.c_void_p]
_Get_Output_To_File.restype  = ctypes.c_bool
def get_output_to_file(p_state):
    """Returns a bool indicating whether the Log is output to a file."""
    return bool(_Get_Output_To_File(ctypes.c_void_p(p_state)))

_Get_Output_File_Level          = _spirit.Log_Get_Output_File_Level
_Get_Output_File_Level.argtypes = [ctypes.c_void_p]
_Get_Output_File_Level.restype  = ctypes.c_int
def get_output_file_level(p_state):
    """Returns the level up to which the Log is output to a file.

    The return value will be one of the integers defined above.
    """
    return int(_Get_Output_File_Level(ctypes.c_void_p(p_state)))