#pragma once
#ifndef SPIRIT_CORE_UTILITY_TIMING_HPP
#define SPIRIT_CORE_UTILITY_TIMING_HPP

#include "Spirit_Defines.h"

#include <chrono>
#include <string>

// Use the System Clock (Wall Time) to handle Timing
using std::chrono::duration;
using std::chrono::system_clock;
using std::chrono::time_point;

namespace Utility
{
namespace Timing
{

std::string TimePointToString( std::chrono::system_clock::time_point t );
std::string TimePointToString_Pretty( std::chrono::system_clock::time_point t );

// Returns the current time like: 2012-05-06_21:47:59
std::string CurrentDateTime();

// Returns the DateTime difference between two DateTimes
std::string DateTimePassed( std::chrono::duration<scalar> dt );

// Returns the difference between two DateTimes in seconds
scalar MillisecondsPassed( std::chrono::duration<scalar> dt );
// Returns the difference between two DateTimes in seconds
scalar SecondsPassed( std::chrono::duration<scalar> dt );
// Returns the difference between two DateTimes in minutes
scalar MinutesPassed( std::chrono::duration<scalar> dt );
// Returns the difference between two DateTimes in hours
scalar HoursPassed( std::chrono::duration<scalar> dt );

// Returns the duration when passed a string "hh:mm:ss"
std::chrono::duration<scalar> DurationFromString( std::string dt );

} // namespace Timing
} // namespace Utility

#endif