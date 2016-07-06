#pragma once
#ifndef UTILITY_LOGGING_H
#define UTILITY_LOGGING_H

#include <iostream>
#include <vector>
#include "Timing.h"

namespace Utility
{

	enum class Log_Sender
	{
		ALL,
		IO,
		GNEB,
		LLG,
		MMF,
		GUI
	};

	enum class Log_Level
	{
		ALL,
		SEVERE,
		L_ERROR,
		WARNING,
		INFO,
		DEBUG
	};

	class LogEntry {
	public:
		// Constructs a log Entry
		LogEntry(system_clock::time_point time, Log_Sender sender, Log_Level level, int image, std::string message);
		// Writes the LogEntry to string
		std::string toString();

		system_clock::time_point time;
		Log_Sender sender;
		Log_Level level;
		int image;
		std::string message;
	};

	class LoggingHandler
	{
	public:
		/*
			Print level defines up to what level the Log is printed into the console.
			All log entries at or above accept level are immediately rejected and not saved.
		*/
		LoggingHandler(Log_Level print_level, Log_Level accept_level, std::string output_folder, std::string fileName);

		// Send a message to the log
		void Send(Log_Level level, Log_Sender s, std::string m);
		void Send(Log_Level level, Log_Sender s, int image, std::string m);

		// Filter the log for debug level, sender and image
		std::vector<LogEntry> Filter(Log_Level level=Log_Level::ALL, Log_Sender sender=Log_Sender::ALL, int image=-1);

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
		// Get the Log's entries
		std::vector<LogEntry> GetEntries();

	private:
		int no_dumped;
		std::vector<LogEntry> log_entries;
	};



	extern LoggingHandler Log;
	extern int logtest;
}

#endif