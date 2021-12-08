#pragma once
#ifndef SPIRIT_CORE_LOG_H
#define SPIRIT_CORE_LOG_H
#include "DLL_Define_Export.h"

#include <vector>

struct State;

/*
Logging
====================================================================

```C
#include "Spirit/Log.h"
```
*/

/*
Definition of log levels and senders
--------------------------------------------------------------------
*/

// Levels
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

// Senders
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

/*
Logging functions
--------------------------------------------------------------------
*/

// Send a Log message
PREFIX void Log_Send(
    State * state, Spirit_Log_Level level, Spirit_Log_Sender sender, const char * message, int idx_image = -1,
    int idx_chain = -1 ) SUFFIX;

// Get the entries from the Log and write new number of entries into given int
// TODO: can this be written in a C-style way?
namespace Utility
{
struct LogEntry;
}
std::vector<Utility::LogEntry> Log_Get_Entries( State * state ) SUFFIX;

// Append the Log to it's file
PREFIX void Log_Append( State * state ) SUFFIX;

// Dump the Log into it's file
PREFIX void Log_Dump( State * state ) SUFFIX;

// Get the number of Log entries
PREFIX int Log_Get_N_Entries( State * state ) SUFFIX;

// Get the number of errors in the Log
PREFIX int Log_Get_N_Errors( State * state ) SUFFIX;

// Get the number of warnings in the Log
PREFIX int Log_Get_N_Warnings( State * state ) SUFFIX;

/*
Set Log parameters
--------------------------------------------------------------------
*/

// The tag in front of the log file
PREFIX void Log_Set_Output_File_Tag( State * state, const char * tag ) SUFFIX;

// The output folder for the log file
PREFIX void Log_Set_Output_Folder( State * state, const char * folder ) SUFFIX;

// Whether to write log messages to the console and corresponding level
PREFIX void Log_Set_Output_To_Console( State * state, bool output, int level ) SUFFIX;

// Whether to write log messages to the log file and corresponding level
PREFIX void Log_Set_Output_To_File( State * state, bool output, int level ) SUFFIX;

/*
Get Log parameters
--------------------------------------------------------------------
*/

// Returns the tag in front of the log file
PREFIX const char * Log_Get_Output_File_Tag( State * state ) SUFFIX;

// Returns the output folder for the log file
PREFIX const char * Log_Get_Output_Folder( State * state ) SUFFIX;

// Returns whether to write log messages to the console
PREFIX bool Log_Get_Output_To_Console( State * state ) SUFFIX;

// Returns the console logging level
PREFIX int Log_Get_Output_Console_Level( State * state ) SUFFIX;

// Returns whether to write log messages to the log file
PREFIX bool Log_Get_Output_To_File( State * state ) SUFFIX;

// Returns the file logging level
PREFIX int Log_Get_Output_File_Level( State * state ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif