#include <utility/Timing.hpp>

#include <string>
#include <iostream>

namespace Utility
{
	namespace Timing
	{
        using namespace std::chrono;
        
        const std::string TimePointToString(system_clock::time_point t)
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
        
        const std::string TimePointToString_Pretty(system_clock::time_point t)
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

        const std::string CurrentDateTime()
        {
            // Get Time Point
            system_clock::time_point now = system_clock::now();
            // Return String from Time Point
			return TimePointToString(now);
		}
        
        
        const std::string DateTimePassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            // Time Difference
            auto dt = system_clock::time_point(t2 - t1);
            // Return String
            return TimePointToString(dt);
        }
        
        const scalar MillisecondsPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1; 
            return  dt.count() * 10e-3;
        }
        
        const scalar SecondsPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1;
            return dt.count();
        }
        
        const scalar MinutesPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1;
			return dt.count() / 60.0;
        }
        
        const scalar HoursPassed(system_clock::time_point t1, system_clock::time_point t2)
        {
            duration<scalar> dt = t2 - t1;
			return dt.count() / 360.0;
        }
    }
}