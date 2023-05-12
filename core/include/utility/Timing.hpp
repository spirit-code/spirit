#pragma once
#ifndef SPIRIT_CORE_UTILITY_TIMING_HPP
#define SPIRIT_CORE_UTILITY_TIMING_HPP

#include <Spirit/Spirit_Defines.h>

#include <chrono>
#include <sstream>
#include <string>

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
std::chrono::duration<scalar> DurationFromString( const std::string & dt );

} // namespace Timing

/**
 * @brief simple std::chrono based timer class
 */
class Timer
{
    using basic_duration_t = std::chrono::microseconds;

    std::chrono::high_resolution_clock::time_point start_, stop_;
    basic_duration_t total_ = basic_duration_t( 0 );
    bool running_           = false;

public:
    void start() noexcept
    {
        if( !running_ )
        {
            running_ = true;
            start_   = std::chrono::high_resolution_clock::now();
        }
    }

    void reset() noexcept
    {
        total_   = basic_duration_t( 0 );
        running_ = false;
    }

    void restart() noexcept
    {
        reset();
        start();
    }

    void stop() noexcept
    {
        if( running_ )
        {
            stop_ = std::chrono::high_resolution_clock::now();
            total_ += std::chrono::duration_cast<basic_duration_t>( stop_ - start_ );
            running_ = false;
        }
    }

    [[nodiscard]] bool running() const noexcept
    {
        return running_;
    }

    template<class Unit>
    [[nodiscard]] Unit elapsed() const noexcept
    {
        return std::chrono::duration_cast<Unit>( current() );
    }

    [[nodiscard]] auto microseconds() const noexcept
    {
        return elapsed<std::chrono::microseconds>().count();
    }

    [[nodiscard]] auto milliseconds() const noexcept
    {
        return elapsed<std::chrono::milliseconds>().count();
    }

    [[nodiscard]] auto full_seconds() const noexcept
    {
        return elapsed<std::chrono::seconds>().count();
    }

    [[nodiscard]] auto full_minutes() const noexcept
    {
        return elapsed<std::chrono::minutes>().count();
    }

    [[nodiscard]] auto full_hours() const noexcept
    {
        return elapsed<std::chrono::hours>().count();
    }

    [[nodiscard]] double seconds() const noexcept
    {
        return ( double( milliseconds() ) / 1000.0 );
    }

    [[nodiscard]] double minutes() const noexcept
    {
        return ( double( milliseconds() ) / 60000.0 );
    }

    [[nodiscard]] double hours() const noexcept
    {
        return ( double( milliseconds() ) / 3600000.0 );
    }

    [[nodiscard]] std::string hh_mm_ss() const noexcept
    {
        std::ostringstream ss;
        int h = static_cast<int>( full_hours() );
        int m = static_cast<int>( full_minutes() );
        int s = static_cast<int>( full_seconds() );
        if( h < 10 )
        {
            ss << "0";
        }
        ss << h << ":";
        if( m < 10 )
        {
            ss << "0";
        }
        ss << m << ":";
        if( s < 10 )
        {
            ss << "0";
        }
        ss << s;
        return ss.str();
    }

private:
    basic_duration_t current() const noexcept
    {
        if( !running_ )
            return total_;

        return (
            total_
            + ( std::chrono::duration_cast<basic_duration_t>( std::chrono::high_resolution_clock::now() - start_ ) ) );
    }
};

} // namespace Utility

// Conversion of time_point to string, usable by fmt
std::ostream & operator<<( std::ostream & os, std::chrono::system_clock::time_point time_point );

#endif
