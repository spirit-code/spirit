#pragma once
#ifndef SPIRIT_CORE_UTILITY_TIMING_HPP
#define SPIRIT_CORE_UTILITY_TIMING_HPP

#include <Spirit/Spirit_Defines.h>

#include <chrono>
#include <string>

namespace Utility
{
namespace Timing
{

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
std::chrono::duration<scalar> DurationFromString( const std::string & dt );

} // namespace Timing
} // namespace Utility

// Conversion of time_point to string, usable by fmt
std::ostream & operator<<( std::ostream & os, std::chrono::system_clock::time_point time_point );

#endif
