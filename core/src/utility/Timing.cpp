#include <utility/Timing.hpp>

#include <string>
#include <iostream>
#include <ctime>

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
            struct tm time_s = *localtime(&t_c);
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
            struct tm time_s = *localtime(&t_c);
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
        
        
        std::string DateTimePassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            // Time Differences
            duration<scalar> _dt = t2 - t1;
            scalar dt = _dt.count();
            int dt_h  = (int)(dt / 3600.0);
            int dt_m  = (int)((dt-3600*dt_h) / 60.0);
            int dt_s  = (int)(dt-3600*dt_h-60*dt_m);
            int dt_ms = (int)((dt-3600*dt_h-60*dt_m-dt_s)*1e3);
            // Return String
            return std::to_string(dt_h) + ":" + std::to_string(dt_m) + ":" + std::to_string(dt_s) + "." + std::to_string(dt_ms);
        }
        
        scalar MillisecondsPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1; 
            return  dt.count() * 10e-3;
        }
        
        scalar SecondsPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1;
            return dt.count();
        }
        
        scalar MinutesPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1;
			return dt.count() / 60.0;
        }
        
        scalar HoursPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1;
			return dt.count() / 3600.0;
        }
    }
}