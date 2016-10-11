#pragma once
#ifndef INTERFACE_LOG_H
#define INTERFACE_LOG_H
#include "DLL_Define_Export.h"

#include <vector>

// Define Log Levels
#define Log_Level_All       0
#define Log_Level_Severe    1
#define Log_Level_Error     2
#define Log_Level_Warning   3
#define Log_Level_Parameter 4
#define Log_Level_Info      5
#define Log_Level_Debug     6

// Define Log Senders
#define Log_Sender_All  0
#define Log_Sender_IO   1
#define Log_Sender_GNEB 2
#define Log_Sender_LLG  3
#define Log_Sender_MMF  4
#define Log_Sender_API  5
#define Log_Sender_UI   6

struct State;

// Send a Log message
DLLEXPORT void Log_Send(State *state, int level, int sender, std::string message, int idx_image=-1, int idx_chain=-1);

// Get the entries from the Log and write new number of entries into given int
// TODO: can this be written in a C-style way?
namespace Utility
{
	struct LogEntry;
}
std::vector<Utility::LogEntry> Log_Get_Entries(State *state);

// Get the number of Log entries
DLLEXPORT int Log_Get_N_Entries(State *state);

// Append the Log to it's file
DLLEXPORT void Log_Append(State *state);

// Dump the Log into it's file
DLLEXPORT void Log_Dump(State *state);

#include "DLL_Undefine_Export.h"
#endif