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
		print_level   = Utility::Log_Level::WARNING;
		accept_level  = Utility::Log_Level::DEBUG;
		output_folder = ".";
		fileName      = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
		n_entries     = 0;
	}

	void LoggingHandler::Send(Log_Level level, Log_Sender sender, std::string message, int idx_image, int idx_chain)
	{
		// All messages are saved in the Log
		LogEntry entry = { std::chrono::system_clock::now(), sender, level, message, idx_image, idx_chain };
		log_entries.push_back(entry);
		
		// If level <= verbosity, we print to console
		if (level <= print_level && level <= accept_level)
			std::cout << LogEntryToString(log_entries.back()) << std::endl;
		
		// Increment message count
		n_entries++;
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
			if (level == Log_Level::ALL || level == entry.level) {
				if (sender == Log_Sender::ALL || sender == entry.sender) {
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
		Send(Log_Level::INFO, Log_Sender::ALL, "Appending Log to file " + output_folder + "/" + fileName);
		
		// Gather the string
		std::string logstring = "";
		int begin_append = no_dumped;
		no_dumped = (int)log_entries.size();
		for (auto entry : log_entries) {
			logstring.append(LogEntryToString(entry));
			logstring.append("\n");
		}

		// Append to file
		IO::Append_String_to_File(logstring, output_folder + "/" + fileName);
	}

	// Write the entire Log to file
	void LoggingHandler::Dump_to_File()
	{
		// Log this event
		Send(Log_Level::INFO, Log_Sender::ALL, "Dumping Log to file " + output_folder + "/" + fileName);

		// Gather the string
		std::string logstring = "";
		no_dumped = (int)log_entries.size();
		for (auto entry : log_entries) {
			logstring.append(LogEntryToString(entry));
			logstring.append("\n");
		}

		// Write the string to file
		IO::String_to_File(logstring, output_folder + "/" + fileName);
	}
}// end namespace Utility