#pragma once
#ifndef UTILITY_EXEPTION_H
#define UTILITY_EXEPTION_H

#include <Spirit/Exception.h>
#include <utility/Logging.hpp>
#include <fmt/format.h>
// #include <sstream>

namespace Utility
{
    enum class Exception_Classifier
    {
        File_not_Found             = Exception_File_not_Found,
        System_not_Initialized     = Exception_System_not_Initialized,
        Division_by_zero           = Exception_Division_by_zero,
        Simulated_domain_too_small = Exception_Simulated_domain_too_small,
        Not_Implemented            = Exception_Not_Implemented,
        Non_existing_Image         = Exception_Non_existing_Image,
        Non_existing_Chain         = Exception_Non_existing_Chain,
        Input_parse_failed,
        Unknown_Exception
        // TODO: from Chain.cpp
        // Last image deletion ?
        // Empty clipboard     ?
    };


    // Spirit library exception class:
    //      Derived from std::runtime_error.
    //      Adds file, line and function information to the exception message.
    //      Contains an exception classifier and a level so that the handler
    //      can decide if execution should be stopped or can continue.
    class S_Exception : public std::runtime_error
    {
    public:
        S_Exception(Exception_Classifier classifier, Log_Level level, const std::string & message, const char * file, unsigned int line, const std::string & function) :
            std::runtime_error(message)
        {
            _message  = fmt::format("{}:{} in function \'{}\': {}", file, line, function, message);
            _file     = file;
            _line     = line;
            _function = function;
            this->classifier = classifier;
            this->level = level;
        }

        ~S_Exception() throw() {}

        const char *what() const throw()
        {
            return _message.c_str();
        }
        
        Exception_Classifier classifier;
        Log_Level level;

    private:
        std::string  _message;
        std::string  _file;
        unsigned int _line;
        std::string  _function;
    };

	// Rethrow (creating a std::nested_exception) an exception using the Exception class
	// to add file and line info
	void rethrow(const std::string & message, const char * file, unsigned int line, const std::string & function);

    // Handle_Exception finalizes what should be done when an exception is encountered.
    //      This function should only be used inside API functions, since that is the
    //      top level at which an exception is caught.
    void Handle_Exception( const std::string & function="", int idx_image=-1, int idx_chain=-1 );


    // Shorthand for throwing a Spirit library exception with file and line info
    #define spirit_throw(classifier, level, message) throw Utility::S_Exception(classifier, level, message, __FILE__, __LINE__, __func__)

	// Rethrow any exception to create a backtraceable nested exception
	#define spirit_rethrow(message) Utility::rethrow(message, __FILE__, __LINE__, __func__)
	
    // Handle exception with backtrace and logging information on the calling API function
    // #define spirit_handle_exception() Handle_Exception(__func__, -1, -1);
    // #define spirit_handle_exception(idx_image) Handle_Exception(__func__, idx_image, -1);
    #define spirit_handle_exception(idx_image, idx_chain) Utility::Handle_Exception(__func__, idx_image, idx_chain)
}

#endif