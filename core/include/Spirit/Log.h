#pragma once
#ifndef INTERFACE_LOG_H
#define INTERFACE_LOG_H
#include "DLL_Define_Export.h"

#include <vector>

// Define Log Levels
typedef enum
{
    Log_Level_All       = 0,
    Log_Level_Severe    = 1,
    Log_Level_Error     = 2,
    Log_Level_Warning   = 3,
    Log_Level_Parameter = 4,
    Log_Level_Info      = 5,
    Log_Level_Debug     = 6
} Spirit_Log_Level;

// Define Log Senders
typedef enum
{
    Log_Sender_All  = 0,
    Log_Sender_IO   = 1,
    Log_Sender_GNEB = 2,
    Log_Sender_LLG  = 3,
    Log_Sender_MC   = 4,
    Log_Sender_MMF  = 5,
    Log_Sender_EMA  = 6,
    Log_Sender_API  = 7,
    Log_Sender_UI   = 8
} Spirit_Log_Sender;

struct State;

//      General functions
// Send a Log message
DLLEXPORT void Log_Send(State *state, Spirit_Log_Level level, Spirit_Log_Sender sender, const char * message, int idx_image=-1, int idx_chain=-1) noexcept;
// Get the entries from the Log and write new number of entries into given int
// TODO: can this be written in a C-style way?
namespace Utility
{
    struct LogEntry;
}
std::vector<Utility::LogEntry> Log_Get_Entries(State *state) noexcept;
// Append the Log to it's file
DLLEXPORT void Log_Append(State *state) noexcept;
// Dump the Log into it's file
DLLEXPORT void Log_Dump(State *state) noexcept;
// Get the number of Log entries
DLLEXPORT int Log_Get_N_Entries(State *state) noexcept;
// Get the number of errors in the Log
DLLEXPORT int Log_Get_N_Errors(State *state) noexcept;
// Get the number of warnings in the Log
DLLEXPORT int Log_Get_N_Warnings(State *state) noexcept;

//      Set Log parameters
DLLEXPORT void Log_Set_Output_File_Tag(State *state, const char * tag) noexcept;
DLLEXPORT void Log_Set_Output_Folder(State *state, const char * folder) noexcept;
DLLEXPORT void Log_Set_Output_To_Console(State *state, bool output, int level) noexcept;
DLLEXPORT void Log_Set_Output_To_File(State *state, bool output, int level) noexcept;

//      Get Log parameters
DLLEXPORT const char * Log_Get_Output_File_Tag(State *state) noexcept;
DLLEXPORT const char * Log_Get_Output_Folder(State *state) noexcept;
DLLEXPORT bool Log_Get_Output_To_Console(State *state) noexcept;
DLLEXPORT int Log_Get_Output_Console_Level(State *state) noexcept;
DLLEXPORT bool Log_Get_Output_To_File(State *state) noexcept;
DLLEXPORT int Log_Get_Output_File_Level(State *state) noexcept;

#include "DLL_Undefine_Export.h"
#endif