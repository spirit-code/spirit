#pragma once
#ifndef IO_FILTERFILEHANDLE_H
#define IO_FILTERFILEHANDLE_H

#include <memory>
#include <string>
#include <istream>
#include <fstream>
#include <sstream>

#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <io/Fileformat.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace IO
{
    class Filter_File_Handle
    {
    private:
        std::size_t found;
        std::string line;
        std::string comment_tag;
        std::string dump;
    public:
        IO::VF_FileFormat ff;
        std::string filename;
        std::unique_ptr<std::ifstream> myfile;
        std::istringstream iss;
        
        // Constructs a Filter_File_Handle with string filename
        Filter_File_Handle( const std::string& s, 
                            IO::VF_FileFormat format = VF_FileFormat::SPIRIT_GENERAL );
        // Destructor
        ~Filter_File_Handle();
        // Reads next line of file into the handle (false -> end-of-file)
        bool GetLine_Handle();
        // Reads the next line of file into the handle and into the iss
        bool GetLine();
        // Reset the file stream to the start of the file
        void ResetStream();
        // Tries to find s in the current file and if found outputs the line into internal iss
        bool Find(const std::string& s);
        // Tries to find s in the current line and if found outputs into internal iss
        bool Find_in_Line(const std::string & s);
        // Removes a set of chars from a string
        void Remove_Chars_From_String(std::string &str, char* charsToRemove);
        // Removes comments from a string
        bool Remove_Comments_From_String( std::string &str );
        // Read a string (separeated by whitespaces) into var
        void Read_String( std::string& var, const std::string keyword, bool log_notfound = true );
        // Count the words of a string
        int Count_Words( const std::string& str );
        // get name 
        
        // Reads a single variable into var, with optional logging in case of failure.
        template <typename T> bool Read_Single( T & var, const std::string name,  
                                                bool log_notfound = true )
        {
            try
            {
                if (Find(name))
                {
                    iss >> var;
                    return true;
                }
                else if (log_notfound)
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO, "Keyword '" + name + 
                        "' not found. Using Default: " + fmt::format( "{}", var ) );
            }
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Failed to read single variable \"{}\".", name));
            }
            return false;
        };
        
        template <typename T> void Require_Single( T& var, const std::string name )
        {
            if( !Read_Single( var, name, false ) )
            {
                spirit_throw(Utility::Exception_Classifier::Bad_File_Content, Utility::Log_Level::Error,
                    fmt::format("Required keyword \"{}\" not found.", name));
            }
        }
        
        // Reads a Vector3 into var, with optional logging in case of failure.
        void Read_Vector3(Vector3 & var, const std::string name, bool log_notfound = true)
        {
            try
            {
                if (Find(name))
                    iss >> var[0] >> var[1] >> var[2];
                else if (log_notfound)
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO, "Keyword '" + name + 
                            "' not found. Using Default: (" + fmt::format( "{}", var.transpose() ) + ")");
            }
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Failed to read Vector3 \"{}\".", name));
            }
        };

        // Reads a 3-component object, with optional logging in case of failure
        template <typename T> void Read_3Vector( T & var, const std::string name, 
                                                    bool log_notfound = true )
        {
            try
            {
                if (Find(name))
                    iss >> var[0] >> var[1] >> var[2];
                else if (log_notfound)
                    Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO, "Keyword '" + name + 
                        "' not found. Using Default: (" + fmt::format( "{}", var[0] ) + " " + 
                        fmt::format( "{}", var[1] ) + " " + fmt::format( "{}", var[2] ) + ")" );

            }
            catch (...)
            {
                spirit_handle_exception_core(fmt::format("Failed to read 3Vector \"{}\".", name));
            }
        };
    };//end class FilterFileHandle
}// end namespace IO

#endif