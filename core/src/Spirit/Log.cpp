#include <Spirit/Log.h>

#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <string>

void Log_Send(
    State * state, Spirit_Log_Level level, Spirit_Log_Sender sender, const char * message, int idx_image,
    int idx_chain ) noexcept
try
{
    Log( static_cast<Utility::Log_Level>( level ), static_cast<Utility::Log_Sender>( sender ), std::string( message ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

std::vector<Utility::LogEntry> Log_Get_Entries( State * state ) noexcept
try
{
    // Get all entries
    return Log.GetEntries();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );

    Utility::LogEntry Error = {
        std::chrono::system_clock::now(),
        Utility::Log_Sender::API,
        Utility::Log_Level::Error,
        { "GetEntries() failed" },
        -1,
        -1,
    };
    std::vector<Utility::LogEntry> ret = { Error };
    return ret;
}

void Log_Append( State * state ) noexcept
try
{
    Log.Append_to_File();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

void Log_Dump( State * state ) noexcept
try
{
    Log.Dump_to_File();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

int Log_Get_N_Entries( State * state ) noexcept
try
{
    return Log.n_entries;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return 0;
}

int Log_Get_N_Errors( State * state ) noexcept
try
{
    return Log.n_errors;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return 0;
}

int Log_Get_N_Warnings( State * state ) noexcept
try
{
    return Log.n_warnings;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return 0;
}

void Log_Set_Output_File_Tag( State * state, const char * tag ) noexcept
try
{
    std::string file_tag = tag;
    Log.file_tag         = file_tag;

    if( file_tag == std::string( "<time>" ) )
        Log.file_name = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
    else if( file_tag == std::string( "" ) )
        Log.file_name = "Log.txt";
    else
        Log.file_name = "Log_" + file_tag + ".txt";
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

void Log_Set_Output_Folder( State * state, const char * folder ) noexcept
try
{
    Log.output_folder = folder;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

void Log_Set_Output_To_Console( State * state, bool output, int level ) noexcept
try
{
    Log.messages_to_console = output;
    Log.level_console       = Utility::Log_Level( level );
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

void Log_Set_Output_To_File( State * state, bool output, int level ) noexcept
try
{
    Log.messages_to_file = output;
    Log.level_file       = Utility::Log_Level( level );
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

const char * Log_Get_Output_File_Tag( State * state ) noexcept
try
{
    return Log.file_tag.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return "";
}

const char * Log_Get_Output_Folder( State * state ) noexcept
try
{
    return Log.output_folder.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return "";
}

bool Log_Get_Output_To_Console( State * state ) noexcept
try
{
    return Log.messages_to_console;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return false;
}

int Log_Get_Output_Console_Level( State * state ) noexcept
try
{
    return (int)Log.level_console;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return 0;
}

bool Log_Get_Output_To_File( State * state ) noexcept
try
{
    return Log.messages_to_file;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return false;
}

int Log_Get_Output_File_Level( State * state ) noexcept
try
{
    return (int)Log.level_file;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return 0;
}