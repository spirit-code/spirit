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
        
        // find begging and end positions of the file stream indicator
        this->position_file_beg = this->myfile->tellg();
        this->myfile->seekg( 0, std::ios::end );
        this->position_file_end = this->myfile->tellg();
        this->myfile->seekg( 0, std::ios::beg );
       
        // set limits of the file stream indicator to begging and end positions (eq. ResetLimits())
        this->position_start = this->position_file_beg;
        this->position_stop = this->position_file_end;
        
        // if the file is not open
        if ( !this->myfile->is_open() )
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
        
        //	if there is a next line
        if ( (bool) getline( *this->myfile, this->line ) )
        {
            //  remove separator characters
            Remove_Chars_From_String( this->line, (char *) "|+" );
            
            // remove any unwanted str from the line eg. delimiters
            if ( str_to_remove != "" )
                Remove_Chars_From_String( this->line, str_to_remove.c_str() );
             
            // if the string does not start with a comment identifier
            if ( Remove_Comments_From_String( this->line ) ) 
                return true;
            else 
                return GetLine();
        }
        return false;     // if there is no next line, return false
    }

    bool Filter_File_Handle::GetLine( const std::string str_to_remove )
    {
        if (Filter_File_Handle::GetLine_Handle( str_to_remove ))
        {
            // decapitalize line
            std::transform( this->line.begin(), this->line.end(), this->line.begin(), ::tolower );
            
            return Filter_File_Handle::Find_in_Line("");
        }
        return false;
    }

    void Filter_File_Handle::ResetStream()
    {
        myfile->clear();
        myfile->seekg(0, std::ios::beg);
    }

    bool Filter_File_Handle::Find(const std::string & s)
    {
        myfile->clear();
        //myfile->seekg( this->position_file_beg, std::ios::beg);
        myfile->seekg( this->position_start );

        while ( GetLine() && ( GetPosition() <= this->position_stop ) ) 
        {
            if (Find_in_Line(s) ) return true;
        }
        return false;
    }

    bool Filter_File_Handle::Find_in_Line( const std::string & s )
    {
        // if s is found in line
        if ( !line.compare( 0, s.size(), s ) )
        {
            iss.clear();    // empty the stream
            iss.str(line);  // copy line into the iss stream
            dump = "";      // TODO: since we have the init in the constructor we might not need that
            
            // if s is not empty
            if ( s.compare("") )
            {
                int words = Count_Words( s );
                for (int i = 0; i < words; i++)
                  iss >> dump;
            }
            
            return true;
        }
        return false;
    }

    void Filter_File_Handle::Remove_Chars_From_String(std::string &str, const char* charsToRemove)
    {
        for (unsigned int i = 0; i < strlen(charsToRemove); ++i)
        {
            str.erase(std::remove(str.begin(), str.end(), charsToRemove[i]), str.end());
        }
    }

    bool Filter_File_Handle::Remove_Comments_From_String( std::string &str )
    {
        std::string::size_type start = this->line.find( this->comment_tag );
        
        // if the line starts with a comment return false
        if ( start == 0 ) return false;
        
        // if the line has a comment somewhere remove it by trimming
        if ( start != std::string::npos )
            line.erase( this->line.begin() + start , this->line.end() );
        
        // return true
        return true;
    }

    void Filter_File_Handle::Read_String( std::string& var, std::string keyword, bool log_notfound )
    {
        std::transform( keyword.begin(), keyword.end(), keyword.begin(), ::tolower );
        
        if ( Find( keyword ) )
        {
            getline( this->iss, var );
            
            // trim leading and trailing whitespaces
            size_t start = var.find_first_not_of(" \t\n\r\f\v");
            size_t end = var.find_last_not_of(" \t\n\r\f\v");
			if ( start != std::string::npos )
				var = var.substr( start, ( end - start + 1 ) );
        }
        else if ( log_notfound )
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::IO,
                 fmt::format( "Keyword \"{}\" not found. Using Default: \"{}\"", keyword, var ) );
    }

    int Filter_File_Handle::Count_Words( const std::string& phrase )
    {
        std::istringstream phrase_stream( phrase );
        this->dump = "";
        int words = 0;
        while( phrase_stream >> dump ) ++words;
        return words; 
    }
}// end namespace IO
