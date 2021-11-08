#pragma once
#ifndef SPIRIT_CORE_UTILITY_EXEPTION_HPP
#define SPIRIT_CORE_UTILITY_EXEPTION_HPP

#include <utility/Logging.hpp>

#include <fmt/format.h>

namespace Utility
{

enum class Exception_Classifier
{
    File_not_Found,
    System_not_Initialized,
    Division_by_zero,
    Simulated_domain_too_small,
    Not_Implemented,
    Non_existing_Image,
    Non_existing_Chain,
    Input_parse_failed,
    Bad_File_Content,
    Standard_Exception,
    CUDA_Error,
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
    S_Exception(
        Exception_Classifier classifier, Log_Level level, const std::string & message, const char * file,
        unsigned int line, const std::string & function )
            : std::runtime_error( message )
    {
        this->classifier = classifier;
        this->level      = level;
        this->message    = message;
        this->file       = file;
        this->line       = line;
        this->function   = function;

        // The complete description
        this->_what = fmt::format( "{}:{} in function \'{}\':\n{:>49}{}", file, line, function, " ", message );
    }

    ~S_Exception() throw() {}

    const char * what() const throw()
    {
        return _what.c_str();
    }

    Exception_Classifier classifier;
    Log_Level level;
    std::string message;
    std::string file;
    unsigned int line;
    std::string function;

private:
    std::string _what;
};

// Rethrow (creating a std::nested_exception) an exception using the Exception class
// to add file and line info
void rethrow( const std::string & message, const char * file, unsigned int line, const std::string & function );

// Handle_Exception_API finalizes what should be done when an exception is encountered at the API layer.
//      This function should only be used inside API functions, since that is the top level at which an
//      exception is caught.
void Handle_Exception_API(
    const char * file, unsigned int line, const std::string & function = "", int idx_image = -1, int idx_chain = -1 );

// Handle_Exception_Core finalizes what should be done when an exception is encountered inside the core.
//      This function should only be used inside the core, below the API layer.
void Handle_Exception_Core( std::string message, const char * file, unsigned int line, const std::string & function );

// Shorthand for throwing a Spirit library exception with file and line info
#define spirit_throw( classifier, level, message )                                                                     \
    throw Utility::S_Exception( classifier, level, message, __FILE__, __LINE__, __func__ )

// Rethrow any exception to create a backtraceable nested exception
#define spirit_rethrow( message ) Utility::rethrow( message, __FILE__, __LINE__, __func__ )

// Handle exception with backtrace and logging information on the calling API function
#define spirit_handle_exception_api( idx_image, idx_chain )                                                            \
    Utility::Handle_Exception_API( __FILE__, __LINE__, __func__, idx_image, idx_chain )

// Handle exception with backtrace and logging information on the calling core function
#define spirit_handle_exception_core( message ) Utility::Handle_Exception_Core( message, __FILE__, __LINE__, __func__ )

} // namespace Utility

#endif