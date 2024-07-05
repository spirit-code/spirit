#include <io/IO.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>

#include <fstream>
#include <memory>
#include <string>

#ifdef CORE_USE_THREADS
#include <thread>
#endif

using Utility::Exception_Classifier;
using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

/*
 * This is a simple RAII file handle that allows streaming strings into a file.
 * Optionally, if CORE_USE_THREADS is defined, it can launch these operations on
 * a new thread.
 */
class OutFileHandle
{
public:
    OutFileHandle( const std::string & filename, bool append )
            : filename( filename ), operation( append ? "appending to" : "writing" )
    {
        if( append )
            myfile.open( filename, std::ofstream::out | std::ofstream::app );
        else
            myfile.open( filename );

        if( !myfile.is_open() )
        {
            spirit_throw(
                Exception_Classifier::File_not_Found, Log_Level::Error,
                fmt::format( "Could not open file \"{}\"", filename ) );
        }
    }

    ~OutFileHandle()
    {
        myfile.close();
    }

    void write( const std::string & str )
    {
        Log( Log_Level::Debug, Log_Sender::All, fmt::format( "Started {} file '{}'", operation, filename ) );
        myfile << str;
        Log( Log_Level::Debug, Log_Sender::All, fmt::format( "Finished {} file '{}'", operation, filename ) );
    }

    void write( const std::vector<std::string> & strings )
    {
        Log( Log_Level::Debug, Log_Sender::All, fmt::format( "Started {} file '{}'", operation, filename ) );
        for( const auto & str : strings )
        {
            myfile << str;
        }
        Log( Log_Level::Debug, Log_Sender::All, fmt::format( "Finished {} file '{}'", operation, filename ) );
    }

private:
    std::string filename;
    std::string operation;
    std::ofstream myfile;
};

void write_to_file( const std::string & str, const std::string & filename )
try
{
    OutFileHandle( filename, false ).write( str );
}
catch( ... )
{
    spirit_handle_exception_core( fmt::format( "Unable to write to file \"{}\"", filename ) );
}

void append_to_file( const std::string & str, const std::string & filename )
try
{
    OutFileHandle( filename, true ).write( str );
}
catch( ... )
{
    spirit_handle_exception_core( fmt::format( "Unable to append to file \"{}\"", filename ) );
}

void dump_to_file( const std::string & str, const std::string & filename )
{
#ifdef CORE_USE_THREADS
    // Fire and forget
    std::thread( write_to_file, str, filename ).detach();
#else
    write_to_file( str, filename );
#endif
}

} // namespace IO
