#pragma once
#ifndef INTERFACE_LOG_H
#define INTERFACE_LOG_H
#include "DLL_Define_Export.h"

struct State;
struct LogEntry;

#include "Logging_Enums.hpp"

#include <iostream>
#include <vector>

// Log Level
using Utility::Log_Level;
//using Utility::Log_Level::All;
//using Utility::Log_Level::Error;
//using Utility::Log_Level::Warning;
//using Utility::Log_Level::Parameter;
//using Utility::Log_Level::Info;
//using Utility::Log_Level::Debug;
// Log Sender - Only UIs will use this
using Utility::Log_Sender;
//using Utility::Log_Sender::UI;
// Log Entry
namespace Utility
{
    struct LogEntry;
}
// using Utility::LogEntry;

// Send a Log message
DLLEXPORT void Log_Send(State *state, Log_Level level, Log_Sender sender, std::string message, int idx_image=-1, int idx_chain=-1);

// Get the entries from the Log and write new number of entries into given int
std::vector<Utility::LogEntry> Log_Get_Entries(State *state);

// Get the number of Log entries
DLLEXPORT int Log_Get_N_Entries(State *state);

// Append the Log to it's file
DLLEXPORT void Log_Append(State *state);

// Dump the Log into it's file
DLLEXPORT void Log_Dump(State *state);

#include "DLL_Undefine_Export.h"
#endif