#pragma once
#ifndef SPIRIT_CORE_IO_FILTERFILEHANDLE_HPP
#define SPIRIT_CORE_IO_FILTERFILEHANDLE_HPP

#include <engine/Vectormath_Defines.hpp>
#include <io/Fileformat.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

namespace IO
{

class Filter_File_Handle
{
private:
    std::size_t found;
    std::string line;
    const std::string comment_tag;
    std::string dump;
    // Beggining and end of file stream indicator
    std::ios::pos_type position_file_beg;
    std::ios::pos_type position_file_end;
    // Start and stop of file stream indicator
    std::ios::pos_type position_start;
    std::ios::pos_type position_stop;
    int n_lines;
    int n_comment_lines;

public:
    std::string filename;
    std::unique_ptr<std::ifstream> myfile;
    std::istringstream iss;

    // Constructs a Filter_File_Handle with string filename
    Filter_File_Handle( const std::string & filename, const std::string comment_tag = "#" );
    // Destructor
    ~Filter_File_Handle();

    // Get the position of the file stream indicator
    std::ios::pos_type GetPosition( std::ios::seekdir dir = std::ios::cur );
    // Set limits in the file stream indicator
    void SetLimits( const std::ios::pos_type beg, const std::ios::pos_type end );
    // Reset the limits of the file stream indicator
    void ResetLimits();
    // Reads next line of file into the handle (false -> end-of-file)
    bool GetLine_Handle( const std::string str_to_remove = "" );
    // Reads the next line of file into the handle and into the iss
    bool GetLine( const std::string str_to_remove = "" );
    // Reset the file stream to the start of the file
    void ResetStream();
    // Tries to find s in the current file and if found outputs the line into internal iss
    bool Find( const std::string & s, bool ignore_case = true );
    // Tries to find s in the current line and if found outputs into internal iss
    bool Find_in_Line( const std::string & s, bool ignore_case = true );
    // Removes a set of chars from a string
    void Remove_Chars_From_String( std::string & str, const char * charsToRemove );
    // Removes comments from a string
    bool Remove_Comments_From_String( std::string & str );
    // Read a string (separeated by whitespaces) into var. Capitalization is ignored.
    void Read_String( std::string & var, std::string keyword, bool log_notfound = true );
    // Count the words of a string
    int Count_Words( const std::string & str );
    // Returns the number of lines which are not starting with a comment
    int Get_N_Non_Comment_Lines();

    // Reads a single variable into var, with optional logging in case of failure.
    //
    //// NOTE: Capitalization is ignored (expected).
    //
    template<typename T>
    bool Read_Single( T & var, std::string keyword, bool log_notfound = true )
    try
    {
        if( Find( keyword ) )
        {
            iss >> var;
            return true;
        }
        else if( log_notfound )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
                 fmt::format( "Keyword '{}' not found. Using Default: {}", keyword, var ) );
        return false;
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format( "Failed to read single variable \"{}\".", keyword ) );
        return false;
    }

    // Require a single field. In case that it is not found an execption is thrown.
    //
    //// NOTE: Capitalization is ignored (expected).
    //
    template<typename T>
    void Require_Single( T & var, std::string keyword )
    {
        if( !Read_Single( var, keyword, false ) )
        {
            spirit_throw(
                Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                fmt::format( "Required keyword \"{}\" not found.", keyword ) );
        }
    }

    // Reads a Vector3 into var, with optional logging in case of failure.
    void Read_Vector3( Vector3 & var, std::string keyword, bool log_notfound = true )
    try
    {
        if( Find( keyword ) )
            iss >> var[0] >> var[1] >> var[2];
        else if( log_notfound )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
                 fmt::format( "Keyword '{}' not found. Using Default: {}", keyword, var.transpose() ) );
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format( "Failed to read Vector3 \"{}\".", keyword ) );
    }

    // Reads a 3-component object, with optional logging in case of failure
    template<typename T>
    void Read_3Vector( T & var, std::string keyword, bool log_notfound = true )
    try
    {
        if( Find( keyword ) )
            iss >> var[0] >> var[1] >> var[2];
        else if( log_notfound )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
                 fmt::format( "Keyword '{}' not found. Using Default: ({} {} {})", keyword, var[0], var[1], var[2] ) );
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format( "Failed to read 3Vector \"{}\".", keyword ) );
    }
};

} // namespace IO

#endif