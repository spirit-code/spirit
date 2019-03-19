#include <io/Filter_File_Handle.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Exception.hpp>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <thread>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace Utility;

namespace IO
{
    Filter_File_Handle::Filter_File_Handle( const std::string& filename,
                                            const std::string comment_tag ) :
        filename(filename), comment_tag(comment_tag), iss("")
    {
        this->dump = "";
        this->line = "";
        this->found = std::string::npos;
        this->myfile = std::unique_ptr<std::ifstream>( new std::ifstream( filename,
                                                        std::ios::in | std::ios::binary ) );
        //this->position = this->myfile->tellg();

        // find begging and end positions of the file stream indicator
        this->position_file_beg = this->myfile->tellg();
        this->myfile->seekg( 0, std::ios::end );
        this->position_file_end = this->myfile->tellg();
        this->myfile->seekg( 0, std::ios::beg );

        // set limits of the file stream indicator to begging and end positions (eq. ResetLimits())
        this->position_start = this->position_file_beg;
        this->position_stop = this->position_file_end;

        // find begging and end positions of the file stream indicator
        this->position_file_beg = this->myfile->tellg();
        this->myfile->seekg( 0, std::ios::end );
        this->position_file_end = this->myfile->tellg();
        this->myfile->seekg( 0, std::ios::beg );

        // set limits of the file stream indicator to begging and end positions (eq. ResetLimits())
        this->position_start = this->position_file_beg;
        this->position_stop = this->position_file_end;

        // initialize number of lines
        this->n_lines = 0;
        this->n_comment_lines = 0;

        // if the file is not open
        if( !this->myfile->is_open() )
        spirit_throw(Exception_Classifier::File_not_Found, Log_Level::Error, fmt::format("Could not open file \"{}\"", filename));
    }

    Filter_File_Handle::~Filter_File_Handle()
    {
        myfile->close();
    }

    std::ios::pos_type Filter_File_Handle::GetPosition( std::ios::seekdir dir )
    {
        this->myfile->seekg( 0, dir );
        return this->myfile->tellg();
    }

    void Filter_File_Handle::SetLimits( const std::ios::pos_type start,
                                        const std::ios::pos_type stop )
    {
        this->position_start = start;
        this->position_stop = stop;
    }

    void Filter_File_Handle::ResetLimits()
    {
        this->position_start = this->position_file_beg;
        this->position_stop = this->position_file_end;
    }

    bool Filter_File_Handle::GetLine_Handle( const std::string str_to_remove )
    {
        this->line = "";

        //  if there is a next line
        if( (bool) getline( *this->myfile, this->line ) )
        {
            this->n_lines++;

            //  remove separator characters
            Remove_Chars_From_String( this->line, (char *) "|+" );

            // remove any unwanted str from the line eg. delimiters
            if( str_to_remove != "" )
                Remove_Chars_From_String( this->line, str_to_remove.c_str() );

            // if the string does not start with a comment identifier
            if( Remove_Comments_From_String( this->line ) )
            {
                return true;
            }
            else
            {
                this->n_comment_lines++;
                return GetLine( str_to_remove );
            }
        }
        return false;     // if there is no next line, return false
    }

    bool Filter_File_Handle::GetLine( const std::string str_to_remove )
    {
        if( Filter_File_Handle::GetLine_Handle(str_to_remove) )
        {
            return Filter_File_Handle::Find_in_Line("");
        }
        return false;
    }

    void Filter_File_Handle::ResetStream()
    {
        myfile->clear();
        myfile->seekg(0, std::ios::beg);
    }

    bool Filter_File_Handle::Find(const std::string & keyword, bool ignore_case)
    {
        myfile->clear();
        //myfile->seekg( this->position_file_beg, std::ios::beg);
        myfile->seekg( this->position_start );

        while( GetLine() && (GetPosition() <= this->position_stop) )
        {
            if( Find_in_Line(keyword, ignore_case) )
                return true;
        }

        return false;
    }

    bool Filter_File_Handle::Find_in_Line( const std::string & keyword, bool ignore_case )
    {
        std::string decap_keyword = keyword;
        std::string decap_line = this->line;
        if( ignore_case )
        {
            std::transform( decap_keyword.begin(), decap_keyword.end(), decap_keyword.begin(), ::tolower );
            std::transform( decap_line.begin(), decap_line.end(), decap_line.begin(), ::tolower );
        }

        // if s is found in line
        if( !decap_line.compare( 0, decap_keyword.size(), decap_keyword ) )
        {
            this->iss.clear();    // empty the stream
            this->iss.str(this->line);  // copy line into the iss stream

            // if s is not empty
            if( keyword != "" )
            {
                int n_words = Count_Words( keyword );
                for( int i = 0; i < n_words; i++ )
                  this->iss >> dump;
            }

            return true;
        }
        return false;
    }

    void Filter_File_Handle::Remove_Chars_From_String(std::string &str, const char* charsToRemove)
    {
        for( unsigned int i = 0; i < strlen(charsToRemove); ++i )
        {
            str.erase(std::remove(str.begin(), str.end(), charsToRemove[i]), str.end());
        }
    }

    bool Filter_File_Handle::Remove_Comments_From_String( std::string &str )
    {
        std::string::size_type start = this->line.find( this->comment_tag );

        // if the line starts with a comment return false
        if( start == 0 ) return false;

        // if the line has a comment somewhere remove it by trimming
        if( start != std::string::npos )
            line.erase( this->line.begin() + start , this->line.end() );

        // return true
        return true;
    }

    void Filter_File_Handle::Read_String( std::string& var, std::string keyword, bool log_notfound )
    {
        if( Find(keyword) )
        {
            getline( this->iss, var );

            // trim leading and trailing whitespaces
            size_t start = var.find_first_not_of(" \t\n\r\f\v");
            size_t end = var.find_last_not_of(" \t\n\r\f\v");
            if( start != std::string::npos )
                var = var.substr( start, ( end - start + 1 ) );
        }
        else if( log_notfound )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
                 fmt::format( "Keyword \"{}\" not found. Using Default: \"{}\"", keyword, var ) );
    }

    int Filter_File_Handle::Count_Words( const std::string& phrase )
    {
        std::istringstream phrase_stream( phrase );
        this->dump = "";
        int words = 0;
        while( phrase_stream >> dump )
            ++words;
        return words;
    }

    int Filter_File_Handle::Get_N_Non_Comment_Lines()
    {
        while( GetLine() ) { };
        ResetLimits();
        return ( this->n_lines - this->n_comment_lines );
    }
}// end namespace IO
