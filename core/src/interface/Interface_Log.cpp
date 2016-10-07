#include "Interface_Log.h"
#include "Logging.hpp"

#include <iostream>

void Log_Send(State *state, int level, int sender, std::string message, int idx_image, int idx_chain)
{
    Log(static_cast<Utility::Log_Level>(level), static_cast<Utility::Log_Sender>(sender), message, idx_image, idx_chain);
}

std::vector<Utility::LogEntry> Log_Get_Entries(State *state)
{
    // Get all entries
    return Log.GetEntries();
}

int Log_Get_N_Entries(State *state)
{
    return Log.n_entries;
}

void Log_Append(State *state)
{
    Log.Append_to_File();
}


void Log_Dump(State *state)
{
    Log.Dump_to_File();
}