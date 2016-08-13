#pragma once
#ifndef INTERFACE_LOG_H
#define INTERFACE_LOG_H
struct State;
struct LogEntry;

#include "Logging_Enums.h"

#include <iostream>
#include <vector>

// Log Level
using Utility::Log_Level;
using Utility::Log_Level::ALL;
using Utility::Log_Level::ERROR;
using Utility::Log_Level::WARNING;
using Utility::Log_Level::PARAMETER;
using Utility::Log_Level::INFO;
using Utility::Log_Level::DEBUG;
// Log Sender - Only UIs will use this
using Utility::Log_Sender;
using Utility::Log_Sender::UI;
// Log Entry
namespace Utility
{
    class LogEntry;
}
// using Utility::LogEntry;

// Send a Log message
extern "C" void Log_Send(State *state, Log_Level level, Log_Sender sender, std::string message, int idx_image=-1, int idx_chain=-1);

// Get the entries from the Log and write new number of entries into given int
std::vector<Utility::LogEntry> Log_Get_Entries(State *state);

// Get the number of Log entries
extern "C" int Log_Get_N_Entries(State *state);

// Append the Log to it's file
extern "C" void Log_Append(State *state);

// Dump the Log into it's file
extern "C" void Log_Dump(State *state);


#endif