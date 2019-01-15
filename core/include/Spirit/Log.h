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
    Log_Sender_UI   = 8,
    Log_Sender_HTST = 9
} Spirit_Log_Sender;

struct State;

//      General functions
// Send a Log message
PREFIX void Log_Send(State *state, Spirit_Log_Level level, Spirit_Log_Sender sender, const char * message, int idx_image=-1, int idx_chain=-1) SUFFIX;
// Get the entries from the Log and write new number of entries into given int
// TODO: can this be written in a C-style way?
namespace Utility
{
    struct LogEntry;
}
std::vector<Utility::LogEntry> Log_Get_Entries(State *state) SUFFIX;
// Append the Log to it's file
PREFIX void Log_Append(State *state) SUFFIX;
// Dump the Log into it's file
PREFIX void Log_Dump(State *state) SUFFIX;
// Get the number of Log entries
PREFIX int Log_Get_N_Entries(State *state) SUFFIX;
// Get the number of errors in the Log
PREFIX int Log_Get_N_Errors(State *state) SUFFIX;
// Get the number of warnings in the Log
PREFIX int Log_Get_N_Warnings(State *state) SUFFIX;

//      Set Log parameters
PREFIX void Log_Set_Output_File_Tag(State *state, const char * tag) SUFFIX;
PREFIX void Log_Set_Output_Folder(State *state, const char * folder) SUFFIX;
PREFIX void Log_Set_Output_To_Console(State *state, bool output, int level) SUFFIX;
PREFIX void Log_Set_Output_To_File(State *state, bool output, int level) SUFFIX;

//      Get Log parameters
PREFIX const char * Log_Get_Output_File_Tag(State *state) SUFFIX;
PREFIX const char * Log_Get_Output_Folder(State *state) SUFFIX;
PREFIX bool Log_Get_Output_To_Console(State *state) SUFFIX;
PREFIX int Log_Get_Output_Console_Level(State *state) SUFFIX;
PREFIX bool Log_Get_Output_To_File(State *state) SUFFIX;
PREFIX int Log_Get_Output_File_Level(State *state) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif