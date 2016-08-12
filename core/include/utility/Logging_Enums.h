#pragma once
#ifndef UTILITY_LOGGING_ENUMS_H
#define UTILITY_LOGGING_ENUMS_H

namespace Utility
{
	enum class Log_Sender
	{
		ALL,
		IO,
		GNEB,
		LLG,
		MMF,
		API,
		UI
	};

	enum class Log_Level
	{
		ALL,
		SEVERE,
		ERROR,
		WARNING,
		PARAMETER,
		INFO,
		DEBUG
	};
}

#endif