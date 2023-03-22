#include <utility/Timing.hpp>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <sstream>
#include <string>

namespace Utility
{
namespace Timing
{

using std::chrono::duration;
using std::chrono::system_clock;

std::string CurrentDateTime()
{
    // Get timepoint
    system_clock::time_point now = system_clock::now();
    // Return string from timepoint
    return fmt::format( "{:%Y-%m-%d_%H-%M-%S}", now );
}

std::string DateTimePassed( duration<scalar> dt )
{
    scalar seconds = dt.count();
    // Time differences
    auto dt_h  = static_cast<int>( seconds / 3600.0 );
    auto dt_m  = static_cast<int>( ( seconds - 3600 * dt_h ) / 60.0 );
    auto dt_s  = static_cast<int>( seconds - 3600 * dt_h - 60 * dt_m );
    auto dt_ms = static_cast<int>( ( seconds - 3600 * dt_h - 60 * dt_m - dt_s ) * 1e3 );
    // Return string
    return fmt::format( "{}:{}:{}.{}", dt_h, dt_m, dt_s, dt_ms );
}

scalar MillisecondsPassed( duration<scalar> dt )
{
    return dt.count() * 1e-3;
}

scalar SecondsPassed( duration<scalar> dt )
{
    return dt.count();
}

scalar MinutesPassed( duration<scalar> dt )
{
    return dt.count() / 60.0;
}

scalar HoursPassed( duration<scalar> dt )
{
    return dt.count() / 3600.0;
}

duration<scalar> DurationFromString( const std::string & dt )
{
    std::int32_t hours = 0, minutes = 0;
    std::int64_t seconds = 0;

    std::istringstream iss( dt );
    std::string token = "";

    // Hours
    if( std::getline( iss, token, ':' ) )
    {
        if( !token.empty() )
            hours = std::stoi( token );
    }

    // Minutes
    if( std::getline( iss, token, ':' ) )
    {
        if( !token.empty() )
            minutes = std::stoi( token );
    }

    // Seconds
    if( std::getline( iss, token, ':' ) )
    {
        if( !token.empty() )
            seconds = std::stol( token );
    }

    // Convert to std::chrono::seconds
    seconds += 60 * minutes + 60 * 60 * hours;

    std::chrono::seconds chrono_seconds( seconds );
    // Return duration
    return duration<scalar>( chrono_seconds );
}

} // namespace Timing
} // namespace Utility
