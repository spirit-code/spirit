#pragma once
#ifndef UTILITY_LOGGING_ENUMS_H
#define UTILITY_LOGGING_ENUMS_H

#include <chrono>
#include <string>

namespace Utility
{
	extern "C" enum class Log_Sender
	{
		ALL,
		IO,
		GNEB,
		LLG,
		MMF,
		API,
		UI
	};

	extern "C" enum class Log_Level
	{
		ALL,
		SEVERE,
		ERROR,
		WARNING,
		PARAMETER,
		INFO,
		DEBUG
	};

	extern "C" struct LogEntry
	{
		std::chrono::system_clock::time_point time;
		Log_Sender sender;
		Log_Level level;
		std::string message;
		int idx_image;
		int idx_chain;
	};

	std::string LogEntryToString(LogEntry entry, bool braces_separators=true);
	
}

#endif