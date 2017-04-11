#include <utility/Logging.hpp>
#include <utility/Timing.hpp>
#include <utility/IO.hpp>

#include <string>
#include <iostream>
#include <ctime>
#include <IO.hpp>
#include <signal.h>


namespace Utility
{
	std::string LogEntryToString(LogEntry entry, bool braces_separators)
	{
		// Format indices
		auto s_chain = IO::int_to_formatted_string(entry.idx_chain+1, 2);
		auto s_image = IO::int_to_formatted_string(entry.idx_image+1, 2);

		// Time
		std::string t = Timing::TimePointToString_Pretty(entry.time);
		std::string result = "";
		result.append(t);
		// Message Level
		if (braces_separators) result.append("  [");
		else result.append("   ");
		if      (entry.level == Log_Level::All)    	result.append("  ALL  ");
		else if (entry.level == Log_Level::Severe) 	result.append("SEVERE ");
		else if (entry.level == Log_Level::Error)	result.append(" ERROR ");
		else if (entry.level == Log_Level::Warning)	result.append("WARNING");
		else if (entry.level == Log_Level::Parameter) result.append(" PARAM ");
		else if (entry.level == Log_Level::Info)    	result.append(" INFO  ");
		else if (entry.level == Log_Level::Debug)   	result.append(" DEBUG ");
		// Sender
		if (braces_separators) result.append("] [");
		else result.append("  ");
		if     (entry.sender == Log_Sender::All)  result.append("ALL ");
		else if(entry.sender == Log_Sender::IO)   result.append("IO  ");
		else if(entry.sender == Log_Sender::API)  result.append("API ");
		else if(entry.sender == Log_Sender::GNEB) result.append("GNEB");
		else if(entry.sender == Log_Sender::LLG)  result.append("LLG ");
		else if(entry.sender == Log_Sender::MMF)  result.append("MMF ");
		else if(entry.sender == Log_Sender::UI)   result.append("UI  ");
		// Chain Index
		if (braces_separators) result.append("] [");
		else result.append("  ");
		if (entry.idx_chain >= 0) result.append(s_chain);
		else result.append("--");
		// Image Index
		if (braces_separators) result.append("] [");
		else result.append("  ");
		if (entry.idx_image >= 0) result.append(s_image);
		else result.append("--");
		if (braces_separators) result.append("]  ");
		else result.append("   ");
		// Message string
		result.append(entry.message);
		// Return
		return result;
	}

	LoggingHandler::LoggingHandler()
	{
		// Set the default Log parameters
		print_level   = Log_Level::Warning;
		accept_level  = Log_Level::Debug;
		output_folder = ".";
		fileName      = "Log_" + Utility::Timing::CurrentDateTime() + ".txt";
		save_output   = false;
		save_input    = true;
		n_entries     = 0;
		n_errors      = 0;
		n_warnings    = 0;
	}

	void LoggingHandler::Send(Log_Level level, Log_Sender sender, std::string message, int idx_image, int idx_chain)
	{
		// Lock mutex because of reallocation (push_back)
		std::lock_guard<std::mutex> guard(mutex);

		// All messages are saved in the Log
		LogEntry entry = { std::chrono::system_clock::now(), sender, level, message, idx_image, idx_chain };
		log_entries.push_back(entry);

		// Increment message count
		n_entries++;
		// Increment error count
		if (level == Log_Level::Error)
			n_errors++;
		// Increment warning count
		if (level == Log_Level::Warning)
			n_warnings++;

		// If level <= verbosity, we print to console
		if (level <= print_level && level <= accept_level)
			std::cout << LogEntryToString(log_entries.back()) << std::endl;
	}

	void LoggingHandler::SendBlock(Log_Level level, Log_Sender sender, std::vector<std::string> messages, int idx_image, int idx_chain)
	{
		// Lock mutex because of reallocation (push_back)
		std::lock_guard<std::mutex> guard(mutex);

		for (auto& message : messages)
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
		if (this->save_output)
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
		else
		{
			Send(Log_Level::Debug, Log_Sender::All, "Not appending Log to file " + output_folder + "/" + fileName);
		}
	}

	// Write the entire Log to file
	void LoggingHandler::Dump_to_File()
	{
		if (this->save_output)
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
		else
		{
			Send(Log_Level::Debug, Log_Sender::All, "Not dumping Log to file " + output_folder + "/" + fileName);
		}
	}
}// end namespace Utility