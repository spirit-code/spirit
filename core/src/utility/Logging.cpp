#include "Logging.h"
#include "Interface_State.h"

#include <string>
#include <iostream>
#include <ctime>
#include <IO.h>
#include <signal.h>

namespace Utility
{
	LoggingHandler::LoggingHandler()
	{
		// Set the default Log parameters
		print_level   = Log_Level::Warning;
		accept_level  = Log_Level::Debug;
		output_folder = ".";
		fileName      = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
		n_entries     = 0;
	}

	void LoggingHandler::Send(Log_Level level, Log_Sender sender, std::string message, int idx_image, int idx_chain)
	{
		// All messages are saved in the Log
		LogEntry entry = { std::chrono::system_clock::now(), sender, level, message, idx_image, idx_chain };
		log_entries.push_back(entry);

		// Increment message count
		n_entries++;

		// If level <= verbosity, we print to console
		if (level <= print_level && level <= accept_level)
			std::cout << LogEntryToString(log_entries.back()) << std::endl;
	}

	void LoggingHandler::operator() (Log_Level level, Log_Sender sender, std::string message, int idx_image, int idx_chain)
	{
		Send(level, sender, message, idx_image, idx_chain);
	}

	std::vector<LogEntry> LoggingHandler::GetEntries()
	{
		return log_entries;
	}

	std::vector<LogEntry> LoggingHandler::Filter(Log_Level level, Log_Sender sender, int idx_image, int idx_chain)
	{
		// Get vector of Log entries
		auto result = std::vector<LogEntry>();
		for (auto entry : log_entries) {
			if (level == Log_Level::All || level == entry.level) {
				if (sender == Log_Sender::All || sender == entry.sender) {
					if (idx_image == -1 || idx_image == entry.idx_image) {
						if (idx_chain == -1 || idx_chain == entry.idx_chain) {
							result.push_back(entry);
						}
					}// endif image no
				}// endif sender
			}// endif level
		}// endfor i -> log_entries.size()

		// Return
		return std::vector<LogEntry>();
	}

	void LoggingHandler::Append_to_File()
	{
		// Log this event
		Send(Log_Level::Info, Log_Sender::All, "Appending Log to file " + output_folder + "/" + fileName);
		
		// Gather the string
		std::string logstring = "";
		int begin_append = no_dumped;
		no_dumped = n_entries;
		for (int i=begin_append; i<n_entries; ++i)
		{
			logstring.append(LogEntryToString(log_entries[i]));
			logstring.append("\n");
		}

		// Append to file
		IO::Append_String_to_File(logstring, output_folder + "/" + fileName);
	}

	// Write the entire Log to file
	void LoggingHandler::Dump_to_File()
	{
		// Log this event
		Send(Log_Level::Info, Log_Sender::All, "Dumping Log to file " + output_folder + "/" + fileName);

		// Gather the string
		std::string logstring = "";
		for (int i=0; i<n_entries; ++i)
		{
			logstring.append(LogEntryToString(log_entries[i]));
			logstring.append("\n");
		}

		// Write the string to file
		IO::String_to_File(logstring, output_folder + "/" + fileName);
	}
}// end namespace Utility