#pragma once
#ifndef UTILITY_IO_FILTERFILEHANDLE_H
#define UTILITY_IO_FILTERFILEHANDLE_H

#include <memory>
#include <string>
#include <istream>
#include <fstream>
#include <sstream>

#include "Logging.hpp"
#include "Vectormath_Defines.hpp"

namespace Utility
{
	namespace IO
	{
		class Filter_File_Handle {
		private:
			std::unique_ptr<std::ifstream> myfile;
			std::string filename;
			std::size_t found;
			std::string line;
			std::string dump;
		public:
			std::istringstream iss;
			// Constructs a Filter_File_Handle with string filename
			Filter_File_Handle(const std::string& s);
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
			// Reads a single variable into var, with logging in case of failure.
			template <typename T> void Read_Single(T & var, const std::string name) {
				if (Find(name)) iss >> var;
				else Log(Utility::Log_Level::Error, Utility::Log_Sender::IO, "Keyword '" + name + "' not found. Using Default: " + stringify(var));
			};
			// Reads a vector into var, with logging in case of failure.
			void Read_Vector3(Vector3 & var, const std::string name) {
				if (Find(name)) iss >> var[0] >> var[1] >> var[2];
				else Log(Utility::Log_Level::Error, Utility::Log_Sender::IO, "Keyword '" + name + "' not found. Using Default: {" + stringify(var[0]) + ", " + stringify(var[1]) + ", " + stringify(var[2]) + "}");
			};
			template <typename T> void Read_3Vector(T & var, const std::string name) {
				if (Find(name)) iss >> var[0] >> var[1] >> var[2];
				else Log(Utility::Log_Level::Error, Utility::Log_Sender::IO, "Keyword '" + name + "' not found. Using Default: {" + stringify(var[0]) + ", " + stringify(var[1]) + ", " + stringify(var[2]) + "}");
			};
			template<class T> typename std::enable_if<std::is_fundamental<T>::value, std::string>::type stringify(const T& t) { return std::to_string(t); }
			template<class T> typename std::enable_if<!std::is_fundamental<T>::value, std::string>::type stringify(const T& t) { return std::string(t); }
		};//end class FilterFileHandle
	}// end namespace IO
}//end namespace Utility

#endif