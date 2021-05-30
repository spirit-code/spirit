#pragma once
#ifndef SPIRIT_UTILITY_LOGGING_HPP
#define SPIRIT_UTILITY_LOGGING_HPP

#include <Spirit/Log.h>
#include <utility/Timing.hpp>

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

// Define Log as the singleton instance, so that messages can be sent with Log(..., message, ...)
#ifndef Log
#define Log Utility::LoggingHandler::getInstance()
#endif

namespace Utility
{

// List of possible senders of a Log Entry
enum class Log_Sender
{
    All  = Log_Sender_All,
    IO   = Log_Sender_IO,
    GNEB = Log_Sender_GNEB,
    LLG  = Log_Sender_LLG,
    MC   = Log_Sender_MC,
    MMF  = Log_Sender_MMF,
    API  = Log_Sender_API,
    UI   = Log_Sender_UI,
    HTST = Log_Sender_HTST,
    EMA  = Log_Sender_EMA
};

// List of possible levels of a Log Entry
enum class Log_Level
{
    All       = Log_Level_All,
    Severe    = Log_Level_Severe,
    Error     = Log_Level_Error,
    Warning   = Log_Level_Warning,
    Parameter = Log_Level_Parameter,
    Info      = Log_Level_Info,
    Debug     = Log_Level_Debug
};

// The Logging Handler contains a vector of Log Entries
struct LogEntry
{
    std::chrono::system_clock::time_point time;
    Log_Sender sender;
    Log_Level level;
    std::vector<std::string> message_lines;
    int idx_image;
    int idx_chain;
};

// Convert the contents of a log entry to a string
std::string LogEntryToString( LogEntry entry, bool braces_separators = true );
// Convert the contents of a log block to a formatted string
std::string LogBlockToString( std::vector<LogEntry> entries, bool braces_separators = true );

/*
The Logging Handler keeps all Log Entries and provides methods to dump or append the entire Log to a file.
Note: the Handler is a singleton.
*/
class LoggingHandler
{
public:
    // Send Log messages
    void Send( Log_Level level, Log_Sender sender, std::string message, int idx_image = -1, int idx_chain = -1 );
    void operator()( Log_Level level, Log_Sender sender, std::string message, int idx_image = -1, int idx_chain = -1 );
    void SendBlock(
        Log_Level level, Log_Sender sender, std::vector<std::string> messages, int idx_image = -1, int idx_chain = -1 );
    void operator()(
        Log_Level level, Log_Sender sender, std::vector<std::string> messages, int idx_image = -1, int idx_chain = -1 );

    // Get the Log's entries
    std::vector<LogEntry> GetEntries();

    // Dumps the log to File fileName
    void Append_to_File();
    void Dump_to_File();

    // The file tag in from of the Log or Output files (if "<time>" is used then the tag is
    // the timestamp)
    std::string file_tag;
    // Output folder where to save the Log file
    std::string output_folder;
    // Save Log messages to file
    bool messages_to_file;
    // All messages up to (including) this level are saved to file
    Log_Level level_file;
    // Print Log messages to console
    bool messages_to_console;
    // All messages up to (including) this level are printed to console
    Log_Level level_console;
    // Save initial input (config / defaults) - note this is done by State_Setup
    bool save_input_initial;
    // Save input at shutdown (config / defaults) - note this is done by State_Delete
    bool save_input_final;
    // Same for positions and neighbours
    bool save_positions_initial;
    bool save_positions_final;
    bool save_neighbours_initial;
    bool save_neighbours_final;
    // Name of the Log file
    std::string fileName;
    // Number of Log entries
    int n_entries;
    // Number of errors in the Log
    int n_errors;
    // Number of warnings in the Log
    int n_warnings;
    // Length of the tags before each message in spaces
    const std::string tags_space = "                                                 ";

    // Retrieve the singleton instance
    static LoggingHandler & getInstance()
    {
        // Guaranteed to be destroyed.
        static LoggingHandler instance;
        // Instantiated on first use.
        return instance;
    }

private:
    // Constructor
    LoggingHandler();

    // Get the Log's entries, filtered for level, sender and indices
    std::vector<LogEntry> Filter(
        Log_Level level = Log_Level::All, Log_Sender sender = Log_Sender::All, int idx_image = -1, int idx_chain = -1 );

    int no_dumped;
    std::vector<LogEntry> log_entries;

    // Mutex for thread-safety
    std::mutex mutex;

public:
    // We can use the better technique of deleting the methods we don't want.
    LoggingHandler( LoggingHandler const & ) = delete;
    void operator=( LoggingHandler const & ) = delete;

    // Note: Scott Meyers mentions in his Effective Modern
    //       C++ book, that deleted functions should generally
    //       be public as it results in better error messages
    //       due to the compilers behavior to check accessibility
    //       before deleted status
};

} // namespace Utility

#endif