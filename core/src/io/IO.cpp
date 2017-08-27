#include <memory>
#include <istream>
#include <fstream>
#include <sstream>
#include <type_traits>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cctype>

#include <io/IO.hpp>
#include <utility/Logging.hpp>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{
    // ------ Formatting Helpers ----------------------------------------

    // Helpers for centering strings
    std::string center(const std::string s, const int w)
    {
        std::stringstream ss, spaces;
        int pad = w - s.size();                  // count excess room to pad
        for(int i=0; i<pad/2; ++i)
            spaces << " ";
        ss << spaces.str() << s << spaces.str(); // format with padding
        if(pad>0 && pad%2!=0)                    // if pad odd #, add 1 more space
            ss << " ";
        return ss.str();
    }

    // trim from start
    static inline std::string &ltrim(std::string &s)
    {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
    }

    // trim from end
    static inline std::string &rtrim(std::string &s)
    {
        s.erase(std::find_if(s.rbegin(), s.rend(),
                std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
    }

    // trim from both ends
    static inline std::string &trim(std::string &s)
    {
        return ltrim(rtrim(s));
    }

    std::string center(const scalar s, const int precision, const int w)
    {
        std::stringstream ss;
        ss << std::setw(w) << std::fixed << std::setprecision(precision) << s;
        std::string ret = ss.str();
        trim(ret);
        return center(ret, w);
    }

    std::string int_to_formatted_string(int in, int n)
    {
        // The format string
        std::string format = "%";
        format += std::to_string(0) + std::to_string(n) + "i";
        // The buffer
        const int buffer_length = 80;
        std::string out = "";
        char buffer_string_conversion[buffer_length + 2];
        //std::cout << format << std::endl;
        // Write formatted into buffer
        snprintf(buffer_string_conversion, buffer_length, format.c_str(), in);
        // Write buffer into out string
        out.append(buffer_string_conversion);
        // Return
        return out;
    }

    // TODO: this function does not make much sense... need to do this stuff coherently throughout the parser...
    std::vector<scalar> split_string_to_scalar(const std::string& source, const std::string& delimiter)
    {
        std::vector<scalar> result;

        scalar temp;
        std::stringstream ss(source);
        while (ss >> temp)
        {
            result.push_back(temp);

            if (ss.peek() == ',' || ss.peek() == ' ')
                ss.ignore();
        }

        return result;
    }

    // ------------------------------------------------------------------


    // ------ Saving Helpers --------------------------------------------

    /*
        Dump_to_File detaches a thread which writes the given string to a file.
        This is asynchronous (i.e. fire & forget)
    */
    void Dump_to_File(const std::string text, const std::string name)
    {
        
        #ifdef CORE_USE_THREADS
        // thread:      method       args  args    args   detatch thread
        std::thread(String_to_File, text, name).detach();
        #else
        String_to_File(text, name);
        #endif
    }

    void Dump_to_File(const std::vector<std::string> text, const std::string name, const int no)
    {
        #ifdef CORE_USE_THREADS
        std::thread(Strings_to_File, text, name, no).detach();
        #else
        Strings_to_File(text, name, no);
        #endif
    }

    /*
        String_to_File is a simple string streamer
        Writing a vector of strings to file
    */
    void Strings_to_File(const std::vector<std::string> text, const std::string name, const int no)
    {

        std::ofstream myfile;
        myfile.open(name);
        if (myfile.is_open())
        {
            Log(Log_Level::Debug, Log_Sender::All, "Started writing " + name);
            for (int i = 0; i < no; ++i) {
                myfile << text[i];
            }
            myfile.close();
            Log(Log_Level::Debug, Log_Sender::All, "Finished writing " + name);
        }
        else
        {
            Log(Log_Level::Error, Log_Sender::All, "Could not open " + name + " to write to file");
        }
    }

    void Append_String_to_File(const std::string text, const std::string name)
    {
        std::ofstream myfile;
        myfile.open(name, std::ofstream::out | std::ofstream::app);
        if (myfile.is_open())
        {
            Log(Log_Level::Debug, Log_Sender::All, "Started writing " + name);
            myfile << text;
            myfile.close();
            Log(Log_Level::Debug, Log_Sender::All, "Finished writing " + name);
        }
        else
        {
            Log(Log_Level::Error, Log_Sender::All, "Could not open " + name + " to write to file");
        }
    }

    void String_to_File(const std::string text, const std::string name)
    {
        std::vector<std::string> v(1);
        v[0] = text;
        Strings_to_File(v, name, 1);
    }

    // ------------------------------------------------------------------
}