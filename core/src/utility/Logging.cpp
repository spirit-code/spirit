#include "Logging.h"
#include "Interface_State.h"

#include <string>
#include <iostream>
#include <ctime>
#include <IO.h>
#include <signal.h>

namespace Utility
{
	LogEntry::LogEntry(system_clock::time_point time, Log_Sender sender, Log_Level level, std::string message, int idx_image, int idx_chain) :
		time(time), sender(sender), level(level), message(message), idx_image(idx_image), idx_chain(idx_chain)
	{
	}

	std::string LogEntry::toString(bool braces_separators)
	{
		// Time
		std::string t = Timing::TimePointToString_Pretty(time);
		std::string result = "";
		result.append(t);
		// Message Level
		if (braces_separators) result.append("  [");
		else result.append("   ");
		if      (level == Log_Level::ALL)    	result.append("  ALL  ");
		else if (level == Log_Level::SEVERE) 	result.append("SEVERE ");
		else if (level == Log_Level::L_ERROR)	result.append(" ERROR ");
		else if (level == Log_Level::WARNING)	result.append("WARNING");
		else if (level == Log_Level::PARAMETER) result.append(" PARAM ");
		else if (level == Log_Level::INFO)    	result.append(" INFO  ");
		else if (level == Log_Level::DEBUG)   	result.append(" DEBUG ");
		// Sender
		if (braces_separators) result.append("] [");
		else result.append("  ");
		if     (sender == Log_Sender::ALL)  result.append("ALL ");
		else if(sender == Log_Sender::IO)   result.append("IO  ");
		else if(sender == Log_Sender::API)  result.append("API ");
		else if(sender == Log_Sender::GNEB) result.append("GNEB");
		else if(sender == Log_Sender::LLG)  result.append("LLG ");
		else if(sender == Log_Sender::MMF)  result.append("MMF ");
		else if(sender == Log_Sender::UI)   result.append("UI  ");
		// Chain Index
		if (braces_separators) result.append("] [");
		else result.append("  ");
		if (idx_chain >= 0) result.append(std::to_string(idx_chain));
		else result.append("--");
		// Image Index
		if (braces_separators) result.append("] [");
		else result.append("  ");
		if (idx_image >= 0) result.append(std::to_string(idx_image));
		else result.append("--");
		if (braces_separators) result.append("]  ");
		else result.append("   ");
		// Message string
		result.append(message);
		// Return
		return result;
	}

	
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
		log_entries.push_back(LogEntry(std::chrono::system_clock::now(), sender, level, message, idx_image, idx_chain));
		
		// If level <= verbosity, we print to console
		if (level <= print_level && level <= accept_level)
			std::cout << log_entries.back().toString() << std::endl;
		
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
			logstring.append(entry.toString());
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
			logstring.append(entry.toString());
			logstring.append("\n");
		}

		// Write the string to file
		IO::String_to_File(logstring, output_folder + "/" + fileName);
	}
}// end namespace Utility