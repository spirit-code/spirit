#pragma once
#ifndef SPIRIT_CORE_UTILITY_EXEPTION_HPP
#define SPIRIT_CORE_UTILITY_EXEPTION_HPP

#include <utility/Logging.hpp>

#include <fmt/format.h>

#include <exception>
#include <type_traits>

namespace Utility
{

enum class Exception_Classifier
{
    // Recoverable
    File_not_Found,
    Division_by_zero,
    Not_Implemented,
    Non_existing_Image,
    Non_existing_Chain,
    Input_parse_failed,
    Bad_File_Content,
    Standard_Exception,
    API_GOT_NULLPTR,
    Unknown_Solver,
    // Unrecoverable
    System_not_Initialized,
    Simulated_domain_too_small,
    CUDA_Error,
    Unknown_Exception
    // TODO: from Chain.cpp
    // Last image deletion ?
    // Empty clipboard     ?
};

/*
 * Spirit library exception class, derived from std::runtime_error.
 *
 * Adds file, line and function information to the exception message. Contains an exception classifier and a level so
 * that the handler can decide if execution should be stopped or can continue.
 */
class Exception : public std::runtime_error
{
public:
    Exception(
        Exception_Classifier classifier, Log_Level level, const std::string & message, const char * file,
        unsigned int line, const char * function ) noexcept( false )
            : std::runtime_error(
                fmt::format( "{}:{} in function \'{}\':\n{:>49}{}", file, line, function, " ", message ) ),
              classifier( classifier ),
              level( level ),
              file( file ),
              line( line ),
              function( function )
    {
    }

    // Copy-constructor
    Exception( const Exception & ) noexcept = default;
    // Copy-assignment operator
    Exception & operator=( const Exception & ) = delete;

    // Move-constructor
    Exception( Exception && ) noexcept = default;
    // Move-assignment constructor
    Exception & operator=( Exception && ) = delete;

    ~Exception() noexcept override = default;

    const Exception_Classifier classifier;
    const Log_Level level;
    const char * file;
    const unsigned int line;
    const char * function;
};

// To make sure that the noexcept-specifications are correct
static_assert(
    std::is_nothrow_copy_constructible<Exception>::value,
    "Utility::Exception is expected to be nothrow copy-constructible" );
static_assert(
    std::is_nothrow_move_constructible<Exception>::value,
    "Utility::Exception is expected to be nothrow move-constructible" );
static_assert(
    std::is_nothrow_destructible<Exception>::value, "Utility::Exception is expected to be nothrow destructible" );

/*
 * Rethrow (creating a std::nested_exception) an exception using the Exception class
 * to add file and line info
 */
void rethrow( const std::string & message, const char * file, const unsigned int line, const char * function ) noexcept(
    false );

/*
 * Handle_Exception_API finalizes what should be done when an exception is encountered at the API layer.
 * This function catches any exceptions and should only be used in the catch-statements of API functions, since that is
 * the top level at which an exception is caught.
 */
void Handle_Exception_API(
    const char * file, const unsigned int line, const char * function = "", const int idx_image = -1,
    const int idx_chain = -1 ) noexcept;

/*
 * Handle_Exception_Core finalizes what should be done when an exception is encountered inside the core library.
 * This function should only be used inside the core, below the API layer.
 */
void Handle_Exception_Core(
    const std::string & message, const char * file, unsigned int line, const char * function ) noexcept( false );

// Shorthand for throwing a Spirit library exception with file and line info
#define spirit_throw( classifier, level, message )                                                                     \
    /* NOLINTNEXTLINE(cppcoreguidelines-macro-usage,hicpp-no-array-decay) */                                           \
    throw Utility::Exception( classifier, level, message, __FILE__, __LINE__, __func__ )

// Rethrow any exception to create a backtraceable nested exception
#define spirit_rethrow( message )                                                                                      \
    /* NOLINTNEXTLINE(cppcoreguidelines-macro-usage,hicpp-no-array-decay) */                                           \
    Utility::rethrow( message, __FILE__, __LINE__, __func__ )

// Handle exception with backtrace and logging information on the calling API function
#define spirit_handle_exception_api( idx_image, idx_chain )                                                            \
    /* NOLINTNEXTLINE(cppcoreguidelines-macro-usage,hicpp-no-array-decay) */                                           \
    Utility::Handle_Exception_API( __FILE__, __LINE__, __func__, idx_image, idx_chain )

// Handle exception with backtrace and logging information on the calling core function
#define spirit_handle_exception_core( message )                                                                        \
    /* NOLINTNEXTLINE(cppcoreguidelines-macro-usage,hicpp-no-array-decay) */                                           \
    Utility::Handle_Exception_Core( message, __FILE__, __LINE__, __func__ )

} // namespace Utility

#endif
