#include <io/IO.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <fmt/format.h>

#include <termcolor/termcolor.hpp>

#include <csignal>
#include <ctime>
#include <iostream>
#include <string>

namespace Utility
{

std::string IndexToString( int idx )
{
    std::string idx_str;
    if( idx >= 0 )
        return fmt::format( "{:0>2}", idx + 1 );
    else
        return "--";
}

std::string LogEntryToString( LogEntry entry, bool braces_separators )
{
    std::string format_str;
    if( braces_separators )
        format_str = "{}  [{:^7}] [{:^4}] [{}] [{}]  {}";
    else
        format_str = "{}   {:^7}   {:^4}   {}   {}   {}";

    std::string result = fmt::format(
        format_str, entry.time, entry.level, entry.sender, IndexToString( entry.idx_chain ),
        IndexToString( entry.idx_image ), entry.message_lines[0] );

    // Rest of the block
    for( std::size_t i = 1; i < entry.message_lines.size(); ++i )
        result += fmt::format( "\n{}{}", Log.tags_space, entry.message_lines[i] );

    return result;
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
    Log_Level level, Log_Sender sender, const std::string & message, int idx_image, int idx_chain )
{
    // Lock mutex because of reallocation (push_back)
    std::lock_guard<std::mutex> guard( mutex );

    // All messages are saved in the Log
    LogEntry entry = { std::chrono::system_clock::now(), sender, level, { message }, idx_image, idx_chain };
    log_entries.push_back( entry );

    // Increment message count
    n_entries++;
    // Increment error count
    if( level == Log_Level::Error )
        n_errors++;
    // Increment warning count
    if( level == Log_Level::Warning )
        n_warnings++;

    // Determine message color in console
    auto color = termcolor::reset;
    if( level <= Log_Level::Warning )
        color = termcolor::yellow;
    if( level <= Log_Level::Error )
        color = termcolor::red;
    if( level == Log_Level::All )
        color = termcolor::reset;

    // If level <= verbosity, we print to console, but Error and Severe are always printed
    if( ( messages_to_console && level <= level_console ) || level == Log_Level::Error || level == Log_Level::Severe )
        std::cout << color << LogEntryToString( log_entries.back() ) << termcolor::reset << "\n";
}

void LoggingHandler::SendBlock(
    Log_Level level, Log_Sender sender, const std::vector<std::string> & messages, int idx_image, int idx_chain )
{
    // Lock mutex because of reallocation (push_back)
    std::lock_guard<std::mutex> guard( mutex );

    // All messages are saved in the Log
    LogEntry entry = { std::chrono::system_clock::now(), sender, level, messages, idx_image, idx_chain };
    log_entries.push_back( entry );

    // Increment message count
    n_entries++;

    // Determine message color in console
    auto color = termcolor::reset;
    if( level <= Log_Level::Warning )
        color = termcolor::yellow;
    if( level <= Log_Level::Error )
        color = termcolor::red;
    if( level == Log_Level::All )
        color = termcolor::reset;

    // If level <= verbosity, we print to console, but Error and Severe are always printed
    if( ( messages_to_console && level <= level_console ) || level == Log_Level::Error || level == Log_Level::Severe )
        std::cout << color << LogEntryToString( log_entries.back() ) << termcolor::reset << "\n";
}

void LoggingHandler::operator()(
    Log_Level level, Log_Sender sender, const std::string & message, int idx_image, int idx_chain )
{
    Send( level, sender, message, idx_image, idx_chain );
}

void LoggingHandler::operator()(
    Log_Level level, Log_Sender sender, const std::vector<std::string> & messages, int idx_image, int idx_chain )
{
    SendBlock( level, sender, messages, idx_image, idx_chain );
}

std::vector<LogEntry> LoggingHandler::GetEntries()
{
    return log_entries;
}

std::vector<LogEntry> LoggingHandler::Filter( Log_Level level, Log_Sender sender, int idx_image, int idx_chain )
{
    // Get vector of Log entries
    auto result = std::vector<LogEntry>();
    for( const auto & entry : log_entries )
    {
        if( ( level == Log_Level::All || level == entry.level )
            && ( sender == Log_Sender::All || sender == entry.sender )
            && ( idx_image == -1 || idx_image == entry.idx_image )
            && ( idx_chain == -1 || idx_chain == entry.idx_chain ) )
        {
            if( sender == Log_Sender::All || sender == entry.sender )
            {
                if( idx_image == -1 || idx_image == entry.idx_image )
                {
                    if( idx_chain == -1 || idx_chain == entry.idx_chain )
                    {
                        result.push_back( entry );
                    }
                }
            }
        }
    }

    // Return
    return result;
}

void LoggingHandler::Append_to_File()
{
    if( this->messages_to_file )
    {
        // Log this event
        Send(
            Log_Level::Debug, Log_Sender::All,
            fmt::format( "Appending log to file \"{}/{}\"", output_folder, file_name ) );

        // Gather the string
        std::string logstring = "";
        int begin_append      = no_dumped;
        no_dumped             = n_entries;
        for( int i = begin_append; i < n_entries; ++i )
        {
            const auto & level = log_entries[i].level;
            if( level <= level_file || level == Log_Level::Error || level == Log_Level::Severe )
            {
                logstring += fmt::format( "{}\n", LogEntryToString( log_entries[i] ) );
            }
        }

        // Append to file
        IO::append_to_file( logstring, output_folder + "/" + file_name );
    }
    else
    {
        Send(
            Log_Level::Debug, Log_Sender::All,
            fmt::format( "Not appending log to file \"{}/{}\"", output_folder, file_name ) );
    }
}

// Write the entire Log to file
void LoggingHandler::Dump_to_File()
{
    if( this->messages_to_file )
    {
        // Log this event
        Send(
            Log_Level::Info, Log_Sender::All,
            fmt::format( "Dumping log to file \"{}/{}\"", output_folder, file_name ) );

        // Gather the string
        std::string logstring = "";
        for( int i = 0; i < n_entries; ++i )
        {
            auto level = log_entries[i].level;
            if( level <= level_file || level == Log_Level::Error || level == Log_Level::Severe )
            {
                logstring += fmt::format( "{}\n", LogEntryToString( log_entries[i] ) );
            }
        }

        // Write the string to file
        IO::write_to_file( logstring, output_folder + "/" + file_name );
    }
    else
    {
        Send(
            Log_Level::Debug, Log_Sender::All,
            fmt::format( "Not dumping log to file \"{}/{}\"", output_folder, file_name ) );
    }
}

} // end namespace Utility

std::ostream & operator<<( std::ostream & os, Utility::Log_Sender sender )
{
    using namespace Utility;

    if( sender == Log_Sender::All )
        return os << "ALL";
    else if( sender == Log_Sender::IO )
        return os << "IO";
    else if( sender == Log_Sender::API )
        return os << "API";
    else if( sender == Log_Sender::GNEB )
        return os << "GNEB";
    else if( sender == Log_Sender::HTST )
        return os << "HTST";
    else if( sender == Log_Sender::LLG )
        return os << "LLG";
    else if( sender == Log_Sender::MC )
        return os << "MC";
    else if( sender == Log_Sender::MMF )
        return os << "MMF";
    else if( sender == Log_Sender::UI )
        return os << "UI";
    else if( sender == Log_Sender::EMA )
        return os << "EMA";
    else
    {
        spirit_throw(
            Exception_Classifier::Not_Implemented, Log_Level::Severe, "Tried converting unknown Log_Sender to string" );
    }
}

// Conversion of Log_Level to string, usable by fmt
std::ostream & operator<<( std::ostream & os, Utility::Log_Level level )
{
    using namespace Utility;

    if( level == Log_Level::All )
        return os << "ALL";
    else if( level == Log_Level::Severe )
        return os << "SEVERE";
    else if( level == Log_Level::Error )
        return os << "ERROR";
    else if( level == Log_Level::Warning )
        return os << "WARNING";
    else if( level == Log_Level::Parameter )
        return os << "PARAM";
    else if( level == Log_Level::Info )
        return os << "INFO";
    else if( level == Log_Level::Debug )
        return os << "DEBUG";
    else
    {
        spirit_throw(
            Exception_Classifier::Not_Implemented, Log_Level::Severe, "Tried converting unknown Log_Level to string" );
    }
}
