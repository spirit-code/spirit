#pragma once
#ifndef UTILITY_LOGGING_H
#define UTILITY_LOGGING_H

#include "Logging_Enums.h"
#include "Timing.h"

#include <iostream>
#include <vector>

#ifndef Log
#define Log Utility::LoggingHandler::getInstance()
#endif

namespace Utility
{
	class LoggingHandler
	{
	public:
		// Send Log messages
		void Send(Log_Level level, Log_Sender sender, std::string message, int idx_image=-1, int idx_chain=-1);
		void operator() (Log_Level level, Log_Sender sender, std::string message, int idx_image=-1, int idx_chain=-1);

		// Get the Log's entries
		std::vector<LogEntry> GetEntries();
		
		// Dumps the log to File fileName
		void Append_to_File();
		void Dump_to_File();

		// All messages up to (including) this level are printed to console
		Log_Level print_level;
		// All log entries at or above reject level are immediately rejected and not saved
		Log_Level accept_level;
		// Output folder where to save the Log file
		std::string output_folder;
		// Name of the Log file
		std::string fileName;
		// Number of Log entries
		int n_entries;

		static LoggingHandler& getInstance()
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
		std::vector<LogEntry> Filter(Log_Level level=Log_Level::ALL, Log_Sender sender=Log_Sender::ALL, int idx_image=-1, int idx_chain=-1);
		
		int no_dumped;
		std::vector<LogEntry> log_entries;
	
	public:
		// C++ 11
		// =======
		// We can use the better technique of deleting the methods
		// we don't want.
		LoggingHandler(LoggingHandler const&)  = delete;
		void operator=(LoggingHandler const&)  = delete;

		// Note: Scott Meyers mentions in his Effective Modern
		//       C++ book, that deleted functions should generally
		//       be public as it results in better error messages
		//       due to the compilers behavior to check accessibility
		//       before deleted status
	};
}

#endif