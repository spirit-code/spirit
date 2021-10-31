#include <io/IO.hpp>
#include <utility/Exception.hpp>
#include <utility/Formatters_Logging.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/format.h>

#include <csignal>
#include <ctime>
#include <iostream>
#include <string>

namespace Utility
{

std::string IndexToString( int idx )
try
{
    std::string idx_str;
    if( idx >= 0 )
        return fmt::format( "{:0>2}", idx + 1 );
    else
        return "--";
}
catch( ... )
{
    spirit_rethrow( "Could not convert index integer to string" );
    return "";
}

LoggingHandler::LoggingHandler()
{
    if( file_tag == "<time>" )
        file_name = fmt::format( "Log_{}.txt", Utility::Timing::CurrentDateTime() );
    else if( file_tag.empty() )
        file_name = "Log.txt";
    else
        file_name = fmt::format( "Log_{}.txt", file_tag );
}

void LoggingHandler::Send(
    Log_Level level, Log_Sender sender, const std::vector<std::string> & messages, int idx_image, int idx_chain )
try
{
    // Lock mutex because of reallocation (push_back)
    std::lock_guard<std::mutex> guard( mutex );

    // All messages are saved in the Log
    LogEntry entry{ std::chrono::system_clock::now(), sender, level, messages, idx_image, idx_chain };
    log_entries.push_back( entry );

    // Increment message count
    n_entries++;
    // Increment error count
    if( level == Log_Level::Error )
        n_errors++;
    // Increment warning count
    if( level == Log_Level::Warning )
        n_warnings++;

    // If level <= verbosity, we print to console, but Error and Severe are always printed
    if( ( messages_to_console && level <= level_console ) || level == Log_Level::Error || level == Log_Level::Severe )
    {
        if( level == Log_Level::Warning )
            fmt::print( fg( fmt::color::orange ) | fmt::emphasis::bold, "{}\n", log_entries.back() );
        else if( level == Log_Level::Error )
            fmt::print( fg( fmt::color::red ) | fmt::emphasis::bold, "{}\n", log_entries.back() );
        else
            fmt::print( "{}\n", log_entries.back() );
    }
}
catch( ... )
{
    spirit_rethrow( "LoggingHandler::SendBlock failed" );
}

void LoggingHandler::operator()(
    Log_Level level, Log_Sender sender, const std::string & message, int idx_image, int idx_chain )
try
{
    Send( level, sender, { message }, idx_image, idx_chain );
}
catch( ... )
{
    spirit_rethrow( "Log(...) call failed" );
}

void LoggingHandler::operator()(
    Log_Level level, Log_Sender sender, const std::vector<std::string> & messages, int idx_image, int idx_chain )
try
{
    Send( level, sender, messages, idx_image, idx_chain );
}
catch( ... )
{
    spirit_rethrow( "Log(...) call failed" );
}

std::vector<LogEntry> LoggingHandler::GetEntries()
try
{
    return log_entries;
}
catch( ... )
{
    spirit_rethrow( "Unable to retrieve log entries" );
    return {};
}

std::vector<LogEntry> LoggingHandler::Filter( Log_Level level, Log_Sender sender, int idx_image, int idx_chain )
try
{
    // Get vector of Log entries
    std::vector<LogEntry> result( 0 );
    for( const auto & entry : log_entries )
    {
        if( ( level == Log_Level::All || level == entry.level )
            && ( sender == Log_Sender::All || sender == entry.sender )
            && ( idx_image == -1 || idx_image == entry.idx_image )
            && ( idx_chain == -1 || idx_chain == entry.idx_chain ) )
        {
            result.push_back( entry );
        }
    }

    // Return
    return result;
}
catch( ... )
{
    spirit_rethrow( "Error filtering the log" );
    return {};
}

void LoggingHandler::Append_to_File()
try
{
    if( this->messages_to_file )
    {
        // Log this event
        Send(
            Log_Level::Debug, Log_Sender::All,
            { fmt::format( "Appending log to file \"{}/{}\"", output_folder, file_name ) } );

        // Gather the string
        std::string logstring = "";
        int begin_append      = n_dumped;
        n_dumped              = n_entries;
        for( int i = begin_append; i < n_entries; ++i )
        {
            const auto & level = log_entries[i].level;
            if( level <= level_file || level == Log_Level::Error || level == Log_Level::Severe )
            {
                logstring += fmt::format( "{}\n", log_entries[i] );
            }
        }

        // Append to file
        IO::append_to_file( logstring, output_folder + "/" + file_name );
    }
    else
    {
        Send(
            Log_Level::Debug, Log_Sender::All,
            { fmt::format( "Not appending log to file \"{}/{}\"", output_folder, file_name ) } );
    }
}
catch( ... )
{
    spirit_rethrow( "Could not append log to file" );
}

// Write the entire Log to file
void LoggingHandler::Dump_to_File()
try
{
    if( this->messages_to_file )
    {
        // Log this event
        Send(
            Log_Level::Info, Log_Sender::All,
            { fmt::format( "Dumping log to file \"{}/{}\"", output_folder, file_name ) } );

        // Gather the string
        std::string logstring = "";
        for( int i = 0; i < n_entries; ++i )
        {
            auto level = log_entries[i].level;
            if( level <= level_file || level == Log_Level::Error || level == Log_Level::Severe )
            {
                logstring += fmt::format( "{}\n", log_entries[i] );
            }
        }

        // Write the string to file
        IO::write_to_file( logstring, output_folder + "/" + file_name );
    }
    else
    {
        Send(
            Log_Level::Debug, Log_Sender::All,
            { fmt::format( "Not dumping log to file \"{}/{}\"", output_folder, file_name ) } );
    }
}
catch( ... )
{
    spirit_rethrow( "Could not dump log to file" );
}

} // end namespace Utility
