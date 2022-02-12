#include <engine/Vectormath.hpp>
#include <io/Filter_File_Handle.hpp>
#include <utility/Exception.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace IO
{

// Removes a set of chars from a string
inline void remove_chars_from_string( std::string & str, const std::string & chars_to_remove )
{
    for( auto c : chars_to_remove )
    {
        str.erase( std::remove( str.begin(), str.end(), c ), str.end() );
    }
}

/*
 * Removes comments from a string
 * Returns false if the string starts with the comment tag. If not, it trims away everything after the the first comment tag
 */
bool remove_comments_from_string( std::string & str, const std::string & comment_tag )
{
    std::string::size_type start = str.find( comment_tag );

    // If the string starts with a comment tag return false
    if( start == 0 )
        return false;

    // If the string has a comment somewhere remove it by trimming
    if( start != std::string::npos )
        str.erase( str.begin() + start, str.end() );

    // Return true
    return true;
}

Filter_File_Handle::Filter_File_Handle( const std::string & filename, const std::string & comment_tag )
        : filename( filename ), comment_tag( comment_tag )
{
    // Open the file
    this->in_file_stream = std::ifstream( filename, std::ios::in | std::ios::binary );

    // Check success
    if( !this->in_file_stream.is_open() )
    {
        spirit_throw(
            Utility::Exception_Classifier::File_not_Found, Utility::Log_Level::Error,
            fmt::format( "Could not open file \"{}\"", filename ) );
    }

    // Find begging and end positions of the file stream indicator
    this->position_file_beg = this->in_file_stream.tellg();
    this->in_file_stream.seekg( 0, std::ios::end );
    this->position_file_end = this->in_file_stream.tellg();
    this->in_file_stream.seekg( 0, std::ios::beg );

    // Set limits of the file stream indicator to begging and end positions (eq. ResetLimits())
    this->position_start = this->position_file_beg;
    this->position_stop  = this->position_file_end;
}

Filter_File_Handle::~Filter_File_Handle()
{
    in_file_stream.close();
}

void Filter_File_Handle::ResetLimits()
{
    this->position_start = this->position_file_beg;
    this->position_stop  = this->position_file_end;
}

bool Filter_File_Handle::GetLine_Handle( const std::string & str_to_remove )
{
    this->current_line = "";

    // If there is a next line
    if( std::getline( this->in_file_stream, this->current_line ) )
    {
        this->n_lines++;

        // Remove separator characters
        remove_chars_from_string( this->current_line, "|+" );

        // Remove any unwanted str from the line eg. delimiters
        if( !str_to_remove.empty() )
            remove_chars_from_string( this->current_line, str_to_remove );

        // If the string does not start with a comment identifier
        if( remove_comments_from_string( this->current_line, this->comment_tag ) )
        {
            return true;
        }
        else
        {
            this->n_comment_lines++;
            return GetLine( str_to_remove );
        }
    }
    return false; // If there is no next line, return false
}

bool Filter_File_Handle::GetLine( const std::string & str_to_remove )
{
    if( Filter_File_Handle::GetLine_Handle( str_to_remove ) )
    {
        return Filter_File_Handle::Find_in_Line( this->current_line, "" );
    }
    return false;
}

void Filter_File_Handle::To_Start()
{
    in_file_stream.clear();
    in_file_stream.seekg( 0, std::ios::beg );
}

bool Filter_File_Handle::Find( const std::string & keyword, const bool ignore_case )
{
    in_file_stream.clear();
    in_file_stream.seekg( this->position_start );

    while( GetLine() && ( this->in_file_stream.tellg() <= this->position_stop ) )
    {
        if( Find_in_Line( this->current_line, keyword, ignore_case ) )
            return true;
    }

    return false;
}

bool Filter_File_Handle::Find_in_Line( const std::string & line, const std::string & keyword, const bool ignore_case )
{
    std::string decap_line    = line;
    std::string decap_keyword = keyword;
    if( ignore_case )
    {
        std::transform( decap_keyword.begin(), decap_keyword.end(), decap_keyword.begin(), ::tolower );
        std::transform( decap_line.begin(), decap_line.end(), decap_line.begin(), ::tolower );
    }

    // If s is found in line
    if( decap_line.compare( 0, decap_keyword.size(), decap_keyword ) == 0 )
    {
        this->iss.clear();     // Empty the stream
        this->iss.str( line ); // Copy line into the iss stream

        if( !keyword.empty() )
        {
            int n_words = Count_Words( keyword );
            for( int i = 0; i < n_words; i++ )
                this->iss >> dump;
        }

        return true;
    }
    return false;
}

void Filter_File_Handle::Read_String( std::string & var, const std::string & keyword, const bool log_notfound )
{
    if( Find( keyword ) )
    {
        getline( this->iss, var );

        // Trim leading and trailing whitespaces
        size_t start = var.find_first_not_of( " \t\n\r\f\v" );
        size_t end   = var.find_last_not_of( " \t\n\r\f\v" );
        if( start != std::string::npos )
            var = var.substr( start, ( end - start + 1 ) );
    }
    else if( log_notfound )
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
             fmt::format( "Keyword \"{}\" not found. Using Default: \"{}\"", keyword, var ) );
}

int Filter_File_Handle::Count_Words( const std::string & str )
{
    std::istringstream stream( str );
    this->dump = "";
    int words  = 0;
    while( stream >> dump )
        ++words;
    return words;
}

int Filter_File_Handle::Get_N_Non_Comment_Lines()
{
    this->n_lines = 0;
    while( GetLine() )
    {
    };
    ResetLimits();
    return ( this->n_lines - this->n_comment_lines );
}

} // namespace IO
