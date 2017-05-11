#include <utility/Timing.hpp>

#include <string>
#include <sstream>
#include <iomanip>


namespace Utility
{
	namespace Timing
	{
        using namespace std::chrono;
        
        std::string TimePointToString(system_clock::time_point t)
        {
            // Convert to C-Time
            std::time_t t_c = system_clock::to_time_t(t);
            // Convert to TM Struct
            std::tm time_s = *localtime(&t_c);
            // Convert TM Struct to String
			char   buf[80];
            strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &time_s);
            // Return
            return buf;
        }
        
        std::string TimePointToString_Pretty(system_clock::time_point t)
        {
            // Convert to C-Time
            std::time_t t_c = system_clock::to_time_t(t);
            // Convert to TM Struct
			std::tm time_s = *localtime(&t_c);
            // Convert TM Struct to String
			char   buf[80];
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &time_s);
            // Return
            return buf;
        }

        std::string CurrentDateTime()
        {
            // Get Time Point
            system_clock::time_point now = system_clock::now();
            // Return String from Time Point
			return TimePointToString(now);
		}
        
        
        std::string DateTimePassed(duration<scalar> _dt)
        {
            // Time Differences
            scalar dt = _dt.count();
            int dt_h  = (int)(dt / 3600.0);
            int dt_m  = (int)((dt-3600*dt_h) / 60.0);
            int dt_s  = (int)(dt-3600*dt_h-60*dt_m);
            int dt_ms = (int)((dt-3600*dt_h-60*dt_m-dt_s)*1e3);
            // Return String
            // return 
			std::string retstring = std::to_string(dt_h) + ":" + std::to_string(dt_m) + ":" + std::to_string(dt_s) + "." + std::to_string(dt_ms);

            return retstring;
        }
        
        scalar MillisecondsPassed(duration<scalar> dt)
        {
            return  dt.count() * 10e-3;
        }
        
        scalar SecondsPassed(duration<scalar> dt)
        {
            return dt.count();
        }
        
        scalar MinutesPassed(duration<scalar> dt)
        {
			return dt.count() / 60.0;
        }
        
        scalar HoursPassed(duration<scalar> dt)
        {
			return dt.count() / 3600.0;
        }


		duration<scalar> DurationFromString(std::string dt)
		{
			// Convert std::string to std::tm
			std::tm tm;
			std::istringstream iss(dt);
			iss >> std::get_time(&tm, "%H:%M:%S");

			// Get total seconds from std::tm
			std::chrono::seconds sec(tm.tm_hour*60*60 + tm.tm_min*60 + tm.tm_sec);

			// Return duration
			return duration<scalar>(sec);
		}
    }
}