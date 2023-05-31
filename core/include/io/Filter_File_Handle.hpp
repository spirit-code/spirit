#pragma once
#ifndef SPIRIT_CORE_IO_FILTERFILEHANDLE_HPP
#define SPIRIT_CORE_IO_FILTERFILEHANDLE_HPP

#include <engine/Vectormath_Defines.hpp>
#include <io/Fileformat.hpp>
#include <utility/Exception.hpp>
#include <utility/Formatters_Eigen.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>

#include <fstream>
#include <sstream>
#include <string>

namespace IO
{

/*
 * RAII file handle, which allows filtering the file contents and reading from it
 */
class Filter_File_Handle
{
public:
    const std::string filename;
    const std::string comment_tag;

    // Constructs a Filter_File_Handle with string filename
    Filter_File_Handle( const std::string & filename, const std::string & comment_tag = "#" );
    // Destructor
    ~Filter_File_Handle();

    // Reset the file stream to the start of the file
    void To_Start();

    // Reads the next line of file into the handle and into the iss
    bool GetLine( const std::string & str_to_remove = "" );
    // Tries to find s in the current file and if found outputs the line into internal iss
    bool Find( const std::string & keyword, bool ignore_case = true );
    // Returns the number of lines which are not starting with a comment
    int Get_N_Non_Comment_Lines();

    // Read a string (separeated by whitespaces) into var. Capitalization is ignored.
    void Read_String( std::string & var, const std::string & keyword, bool log_notfound = true );

    /**
     * Read a single variable into var, with optional logging in case of failure.
     *
     * NOTE: Capitalization is ignored (expected).
     */
    template<typename T>
    bool Read_Single( T & var, const std::string & keyword, const bool log_notfound = true )
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

    /**
     * Require a single field. In case that it is not found an execption is thrown.
     *
     * NOTE: Capitalization is ignored (expected).
     */
    template<typename T>
    void Require_Single( T & var, const std::string & keyword )
    {
        if( !Read_Single( var, keyword, false ) )
        {
            spirit_throw(
                Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                fmt::format( "Required keyword \"{}\" not found.", keyword ) );
        }
    }

    // Reads a Vector3 into var, with optional logging in case of failure.
    void Read_Vector3( Vector3 & var, const std::string & keyword, const bool log_notfound = true )
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
    void Read_3Vector( T & var, const std::string & keyword, const bool log_notfound = true )
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

private:
    std::ifstream in_file_stream;
    std::istringstream iss{ "" };

    std::string current_line{ "" };
    std::string dump{ "" };

    int n_lines{ 0 };
    int n_comment_lines{ 0 };

    // Beggining and end of file stream indicator
    std::ios::pos_type position_file_beg;
    std::ios::pos_type position_file_end;

    // Start and stop of file stream indicator
    std::ios::pos_type position_start;
    std::ios::pos_type position_stop;

    // Reset the limits of the file stream indicator
    void ResetLimits();
    // Reads next line of file into the handle (false -> end-of-file)
    bool GetLine_Handle( const std::string & str_to_remove = "" );
    // Tries to find s in the current line and if found outputs into internal iss
    bool Find_in_Line( const std::string & line, const std::string & keyword, bool ignore_case = true );
    // Count the words of a string
    int Count_Words( const std::string & str );

    // Extract something from the file
    template<typename T>
    friend std::istringstream & operator>>( Filter_File_Handle & file, T & obj )
    {
        file.iss >> obj;
        return file.iss;
    }
};

} // namespace IO

#endif
