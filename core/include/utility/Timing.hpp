#pragma once
#ifndef UTILITY_TIMING_H
#define UTILITY_TIMING_H

#include <string>
#include <chrono>

#include "Spirit_Defines.h"

// Use the System Clock (Wall Time) to handle Timing
using std::chrono::system_clock;
using std::chrono::time_point;
using std::chrono::duration;

namespace Utility
{
    namespace Timing
    {   
        std::string TimePointToString(system_clock::time_point t);
        std::string TimePointToString_Pretty(system_clock::time_point t);
        
		// Returns the current time like: 2012-05-06_21:47:59
		std::string CurrentDateTime();
        
        // Returns the DateTime difference between two DateTimes
		std::string DateTimePassed(duration<scalar> dt);
        
        // Returns the difference between two DateTimes in seconds
        scalar MillisecondsPassed(duration<scalar> dt);
        // Returns the difference between two DateTimes in seconds
		scalar SecondsPassed(duration<scalar> dt);
        // Returns the difference between two DateTimes in minutes
        scalar MinutesPassed(duration<scalar> dt);
        // Returns the difference between two DateTimes in hours
        scalar HoursPassed(duration<scalar> dt);
        
        // Returns the duration when passed a string "hh:mm:ss"
		duration<scalar> DurationFromString(std::string dt);
    }
}
#endif