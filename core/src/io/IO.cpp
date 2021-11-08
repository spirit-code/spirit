#include <io/IO.hpp>
#include <utility/Logging.hpp>

#include <fstream>
#include <memory>
#include <string>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

/*
Dump_to_File detaches a thread which writes the given string to a file.
This is asynchronous (i.e. fire & forget)
*/
void Dump_to_File( const std::string text, const std::string name )
{
#ifdef CORE_USE_THREADS
    // thread:      method       args  args    args   detatch thread
    std::thread( String_to_File, text, name ).detach();
#else
    String_to_File( text, name );
#endif
}

void Dump_to_File( const std::vector<std::string> text, const std::string name, const int no )
{
#ifdef CORE_USE_THREADS
    std::thread( Strings_to_File, text, name, no ).detach();
#else
    Strings_to_File( text, name, no );
#endif
}

/*
String_to_File is a simple string streamer
Writing a vector of strings to file
*/
void Strings_to_File( const std::vector<std::string> text, const std::string name, const int no )
{
    std::ofstream myfile;
    myfile.open( name );
    if( myfile.is_open() )
    {
        Log( Log_Level::Debug, Log_Sender::All, "Started writing " + name );
        for( int i = 0; i < no; ++i )
        {
            myfile << text[i];
        }
        myfile.close();
        Log( Log_Level::Debug, Log_Sender::All, "Finished writing " + name );
    }
    else
    {
        Log( Log_Level::Error, Log_Sender::All, "Could not open " + name + " to write to file" );
    }
}

void Append_String_to_File( const std::string text, const std::string name )
{
    std::ofstream myfile;
    myfile.open( name, std::ofstream::out | std::ofstream::app );
    if( myfile.is_open() )
    {
        Log( Log_Level::Debug, Log_Sender::All, "Started writing " + name );
        myfile << text;
        myfile.close();
        Log( Log_Level::Debug, Log_Sender::All, "Finished writing " + name );
    }
    else
    {
        Log( Log_Level::Error, Log_Sender::All, "Could not open " + name + " to append to file" );
    }
}

void String_to_File( const std::string text, const std::string name )
{
    std::vector<std::string> v( 1 );
    v[0] = text;
    Strings_to_File( v, name, 1 );
}

} // namespace IO