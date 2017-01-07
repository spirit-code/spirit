#include <utility/IO_Filter_File_Handle.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <engine/Vectormath.hpp>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <thread>
#include <string>
#include <cstring>
#include <sstream>

namespace Utility
{
	namespace IO
	{
		Filter_File_Handle::Filter_File_Handle(const std::string& filename):
			filename(filename), iss("")
		{
			dump = "";
			line = "";
			found = std::string::npos;
			myfile = std::unique_ptr<std::ifstream>(new std::ifstream(filename));
			if (!myfile->is_open()) { throw Utility::Exception::File_not_Found; }
			else {
				Log(Log_Level::Debug, Utility::Log_Sender::IO, std::string("  Reading from Config File ").append(filename));
			}
		}// end constructor

		Filter_File_Handle::~Filter_File_Handle()
		{ 
			myfile->close();
			Log(Log_Level::Debug, Utility::Log_Sender::IO, std::string("  Config File Closed: ").append(filename));
		}// end destructor

		bool Filter_File_Handle::GetLine_Handle()
		{
			line = "";
			if ((bool)getline(*myfile, line))
			{													//	if there is a next line
				Remove_Chars_From_String(line, (char*)"|+");	//  remove separator characters
				found = line.find("#");							//	try to find #
				if (found == std::string::npos)	return true;	//	if # not found -> return true
				else return GetLine();							//	else return GetLine(s) to read next line
			}// endif file-not-ended
			return false;										//	if there is no next line, return false
		}

		bool Filter_File_Handle::GetLine()
		{
			if (Filter_File_Handle::GetLine_Handle()) {
				return Filter_File_Handle::Find_in_Line("");
			}
			return false;
		}
		// end Filter_File_Handle::GetLine()

		void Filter_File_Handle::ResetStream()
		{
			myfile->clear();
			myfile->seekg(0, std::ios::beg);
		}

		bool Filter_File_Handle::Find(const std::string & s)
		{
			myfile->clear();
			myfile->seekg(0, std::ios::beg);

			while (GetLine())
			{
				if (Find_in_Line(s)) return true;
			}
			return false;
		}
		// end Filter_File_Handle::Find()

		bool Filter_File_Handle::Find_in_Line(const std::string & s)
		{
			if (!line.compare(0, s.size(), s)) {				// if s is found in line
			// if(line.compare(s))  {
				iss.clear();
				iss.str(line);									// copy line into the iss stream
				dump = "";
				if (s.compare("")) {							// if s is not empty
					iss >> dump;								// dump the first entry of the stream
				}// endif
				return true;
			}// endif s is found in line
			return false;
		}// end Filter_File_Handle::Find()


		void Filter_File_Handle::Remove_Chars_From_String(std::string &str, char* charsToRemove)
		{
			for (unsigned int i = 0; i < strlen(charsToRemove); ++i) {
				str.erase(std::remove(str.begin(), str.end(), charsToRemove[i]), str.end());
			}
		}
	}// end namespace IO
}// end namespace Utility