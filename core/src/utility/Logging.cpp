#include "Logging.h"
#include "Interface_State.h"

#include <string>
#include <iostream>
#include <ctime>
#include <IO.h>
#include <signal.h>

namespace Utility
{
	LogEntry::LogEntry(system_clock::time_point t, Log_Sender s, Log_Level l, int i, std::string m) :
		time(t), sender(s), level(l), image(i), message(m)
	{
	}

	std::string LogEntry::toString()
	{
		std::string t = Timing::TimePointToString(time);
		std::string result = "[";
		result.append(t);
		result.append("] [");
		if     (sender == Log_Sender::ALL)  result.append("ALL ");
		else if(sender == Log_Sender::IO)   result.append(" IO ");
		else if(sender == Log_Sender::API)  result.append("API ");
		else if(sender == Log_Sender::GNEB) result.append("GNEB");
		else if(sender == Log_Sender::LLG)  result.append("LLG ");
		else if(sender == Log_Sender::MMF)  result.append("MMF ");
		else if(sender == Log_Sender::UI)  result.append(" UI ");
		result.append("] [");
		if      (level == Log_Level::ALL)     result.append("  ALL  ");
		else if (level == Log_Level::SEVERE)  result.append("SEVERE ");
		else if (level == Log_Level::L_ERROR) result.append(" ERROR ");
		else if (level == Log_Level::WARNING) result.append("WARNING");
		else if (level == Log_Level::INFO)    result.append(" INFO  ");
		else if (level == Log_Level::DEBUG)   result.append(" DEBUG ");
		result.append("] [");
		result.append(std::to_string(image));
		result.append("] : ");
		result.append(message);
		return result;
	}

	LoggingHandler::LoggingHandler(Log_Level print_level_i, Log_Level accept_level_i, std::string output_folder_i, std::string fileName_i) :
		print_level(print_level_i), accept_level(accept_level_i), output_folder(output_folder_i), fileName(fileName_i), n_entries(0)
	{
	}

	// Send a Log message without image index
	void LoggingHandler::Send(Log_Level level, Log_Sender s, std::string m)
	{
		Send(level, s, -1, m);
	}

	// Send a Log message
	void LoggingHandler::Send(Log_Level level, Log_Sender s, int image, std::string m)
	{
		// All messages are saved in the Log
		log_entries.push_back(LogEntry(std::chrono::system_clock::now(), s, level, image, m));
		
		// If level <= verbosity, we print to console
		if (level <= print_level && level <= accept_level)
			std::cout << log_entries[(int)log_entries.size() - 1].toString() << std::endl;
		
		n_entries++;
	}

	// Filter the messages saved in the Log by levels and/or senders and/or image index
	std::vector<LogEntry> LoggingHandler::Filter(Log_Level level, Log_Sender sender, int image)
	{
		auto result = std::vector<LogEntry>();
		for (unsigned int i = 0; i < log_entries.size(); ++i) {
			if (level == Log_Level::ALL || level == log_entries[i].level) {
				if (sender == Log_Sender::ALL || sender == log_entries[i].sender) {
					if (image == -1 || image == log_entries[i].image) {
						result.push_back(log_entries[i]);
					}// endif image no
				}// endif sender
			}// endif level
		}// endfor i -> log_entries.size()
		return std::vector<LogEntry>();
	}

	void LoggingHandler::Append_to_File()
	{
		Send(Log_Level::INFO, Log_Sender::ALL, "Appending Log to file " + Log.output_folder + "/" + Log.fileName);
		std::string logstring = "";
		int begin_append = no_dumped;
		no_dumped = (int)log_entries.size();
		for (int i = begin_append; i < no_dumped; ++i) {
			logstring.append(log_entries[i].toString());
			logstring.append("\n");
		}
		IO::Append_String_to_File(logstring, Log.output_folder + "/" + Log.fileName);
	}

	// Write the entire Log to file
	void LoggingHandler::Dump_to_File()
	{
		Send(Log_Level::INFO, Log_Sender::ALL, "Dumping Log to file " + Log.output_folder + "/" + Log.fileName);
		std::string logstring = "";
		no_dumped = (int)log_entries.size();
		for (int i = 0; i < no_dumped; ++i) {
			logstring.append(log_entries[i].toString());
			logstring.append("\n");
		}
		IO::String_to_File(logstring, Log.output_folder + "/" + Log.fileName);
	}

	std::vector<LogEntry> LoggingHandler::GetEntries()
	{
		return this->log_entries;
	}

	
}// end namespace Utility