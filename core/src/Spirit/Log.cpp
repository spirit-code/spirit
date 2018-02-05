#include <Spirit/Log.h>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <string>

void Log_Send( State *state, int level, int sender, const char * message, int idx_image, int idx_chain ) noexcept
{
    try
    {
        Log( static_cast<Utility::Log_Level>(level), static_cast<Utility::Log_Sender>(sender), 
             std::string(message), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

std::vector<Utility::LogEntry> Log_Get_Entries(State *state) noexcept
{
    try
    {
        // Get all entries
        return Log.GetEntries();
    }
    catch( ... )
    {
        spirit_handle_exception_api(-1, -1);
        
        Utility::LogEntry Error = { std::chrono::system_clock::now(), 
                                    Utility::Log_Sender::API, Utility::Log_Level::Error,
                                    "GetEntries() failed", -1, -1 };
        std::vector<Utility::LogEntry> ret = { Error };
        return ret;
    }
}

void Log_Append(State *state) noexcept
{
    try
    {
        Log.Append_to_File();
    }
    catch( ... )
    {
        spirit_handle_exception_api(-1, -1);
    }
}

void Log_Dump(State *state) noexcept
{
    try
    {
        Log.Dump_to_File();
    }
    catch( ... )
    {
        spirit_handle_exception_api(-1, -1);
    }
}

int Log_Get_N_Entries(State *state) noexcept
{
    return Log.n_entries;
}

int Log_Get_N_Errors(State *state) noexcept
{
    return Log.n_errors;
}

int Log_Get_N_Warnings(State *state) noexcept
{
    return Log.n_warnings;
}