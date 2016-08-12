#include "Interface_Log.h"
#include "Logging.h"

extern "C" void Log_Send(State *state, Log_Level level, Log_Sender sender, std::string message, int idx_image, int idx_chain)
{
    Log(level, sender, message, idx_image, idx_chain);
}

std::vector<Utility::LogEntry> Log_Get_Entries(State *state)
{
    // Get all entries
    return Log.GetEntries();
}

extern "C" void Log_Append(State *state)
{
    Log.Append_to_File();
}


extern "C" void Log_Dump(State *state)
{
    Log.Dump_to_File();
}