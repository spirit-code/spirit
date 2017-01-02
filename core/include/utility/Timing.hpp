#pragma once
#ifndef UTILITY_TIMING_H
#define UTILITY_TIMING_H

#include <string>
#include <ctime>
#include <chrono>

#include "Core_Defines.h"

// Use the System Clock (Wall Time) to handle Timing
using std::chrono::system_clock;
using std::chrono::time_point;

namespace Utility
{
    namespace Timing
    {   
        std::string TimePointToString(system_clock::time_point t);
        std::string TimePointToString_Pretty(system_clock::time_point t);
        
		// Returns the current time like: 2012-05-06_21:47:59
		std::string CurrentDateTime();
        
        // Returns the DateTime difference between two DateTimes
        std::string DateTimePassed(system_clock::time_point t1, system_clock::time_point t2);
        
        // Returns the difference between two DateTimes in seconds
        scalar MillisecondsPassed(system_clock::time_point t1, system_clock::time_point t2);
        // Returns the difference between two DateTimes in seconds
		scalar SecondsPassed(system_clock::time_point t1, system_clock::time_point t2);
        // Returns the difference between two DateTimes in minutes
        scalar MinutesPassed(system_clock::time_point t1, system_clock::time_point t2);
        // Returns the difference between two DateTimes in hours
        scalar HoursPassed(system_clock::time_point t1, system_clock::time_point t2);
    }
}
#endif