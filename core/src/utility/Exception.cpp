#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>

namespace Utility
{

void rethrow( const std::string & message, const char * file, const unsigned int line, const char * function ) noexcept(
    false )
try
{
    std::rethrow_exception( std::current_exception() );
}
catch( const Exception & ex )
{
    auto ex2 = Exception( ex.classifier, ex.level, message, file, line, function );
    std::throw_with_nested( ex2 );
}
catch( const std::exception & ex )
{
    auto ex2 = Exception( Exception_Classifier::Standard_Exception, Log_Level::Severe, message, file, line, function );
    std::throw_with_nested( ex2 );
}
catch( ... )
{
    auto ex = Exception( Exception_Classifier::Unknown_Exception, Log_Level::Severe, message, file, line, function );
    std::throw_with_nested( ex );
}

void Backtrace_Exception() noexcept( false )
try
{
    if( std::exception_ptr eptr = std::current_exception() )
    {
        std::rethrow_exception( eptr );
    }
    else
    {
        Log( Log_Level::Severe, Log_Sender::API, "Unknown Exception. Terminating" );
        Log.Append_to_File();
        std::exit( EXIT_FAILURE ); // Exit the application. May lead to data loss!
    }
}
catch( const Exception & ex )
{
    Log( ex.level, Log_Sender::API, std::string( ex.what() ) );
    try
    {
        rethrow_if_nested( ex );
    }
    catch( ... )
    {
        Backtrace_Exception();
    }
}
catch( const std::exception & ex )
{
    Log( Log_Level::Severe, Log_Sender::API, fmt::format( "std::exception: \"{}\"", ex.what() ) );
    try
    {
        rethrow_if_nested( ex );
    }
    catch( ... )
    {
        Backtrace_Exception();
    }
}

void Handle_Exception_API(
    const char * file, const unsigned int line, const char * function, const int idx_image,
    const int idx_chain ) noexcept
try
{
    // Rethrow in order to get an exception reference (instead of pointer)
    try
    {
        std::rethrow_exception( std::current_exception() );
    }
    catch( const Exception & ex )
    {
        Log( ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain );
        std::string str_exception;
        if( int( ex.level ) > 1 )
            str_exception = "exception";
        else
            str_exception = "SEVERE exception";
        Log( ex.level, Log_Sender::API,
             fmt::format(
                 "Caught {} in API function \'{}\' (at {}:{})\n{}Exception backtrace:", str_exception, function, file,
                 line, Log.tags_space ),
             idx_image, idx_chain );

        // Create backtrace
        Backtrace_Exception();
        Log( ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain );

        // Check if the exception was recoverable
        if( ex.classifier == Exception_Classifier::Unknown_Exception
            || ex.classifier == Exception_Classifier::System_not_Initialized
            || ex.classifier == Exception_Classifier::Simulated_domain_too_small
            || ex.classifier == Exception_Classifier::CUDA_Error || int( ex.level ) <= 1 )
        {
            Log( Log_Level::Severe, Log_Sender::API, "TERMINATING!", idx_image, idx_chain );
            Log.Append_to_File();
            std::exit( EXIT_FAILURE ); // Exit the application. May lead to data loss!
        }

        // If it was recoverable we now write to Log
        Log.Append_to_File();
    }
    catch( const std::exception & ex )
    {
        Log( Log_Level::Severe, Log_Sender::API, "-----------------------------------------------------", idx_image,
             idx_chain );
        Log( Log_Level::Severe, Log_Sender::API,
             fmt::format(
                 "Caught std::exception in API function \'{}\' (at {}:{})\n{:>49}Exception backtrace:", function, file,
                 line, " " ),
             idx_image, idx_chain );
        // Create backtrace
        Backtrace_Exception();
        Log( Log_Level::Severe, Log_Sender::API, "-----------------------------------------------------", idx_image,
             idx_chain );

        // Terminate
        Log( Log_Level::Severe, Log_Sender::API, "TERMINATING!", idx_image, idx_chain );
        Log.Append_to_File();
        std::exit( EXIT_FAILURE ); // Exit the application. May lead to data loss!
    }
    catch( ... )
    {
        Log( Log_Level::Severe, Log_Sender::API, "-----------------------------------------------------", idx_image,
             idx_chain );
        Log( Log_Level::Severe, Log_Sender::API,
             fmt::format(
                 "Caught unknown exception in API function \'{}\' (at {}:{})\n{:>49}Cannot backtrace unknown "
                 "exceptions...",
                 function, file, line, " " ),
             idx_image, idx_chain );
        Log( Log_Level::Severe, Log_Sender::API, "-----------------------------------------------------", idx_image,
             idx_chain );
        Log( Log_Level::Severe, Log_Sender::API, "Unable to handle! TERMINATING!", idx_image, idx_chain );
        Log.Append_to_File();
        std::exit( EXIT_FAILURE ); // Exit the application. May lead to data loss!
    }
}
catch( ... )
{
    std::cerr << "Another exception occurred while handling an exception from \'" << function << "\' (at " << file
              << ":" << line << ")!" << std::endl;
    std::cerr << "TERMINATING!" << std::endl;
    std::exit( EXIT_FAILURE ); // Exit the application. May lead to data loss!
}

void Handle_Exception_Core(
    const std::string & message, const char * file, const unsigned int line, const char * function ) noexcept( false )
try
{
    // Rethrow in order to get an exception reference (instead of pointer)
    std::rethrow_exception( std::current_exception() );
}
catch( const Exception & ex )
{
    bool can_handle = true;
    if( ex.classifier == Exception_Classifier::Unknown_Exception
        || ex.classifier == Exception_Classifier::System_not_Initialized
        || ex.classifier == Exception_Classifier::Simulated_domain_too_small
        || ex.classifier == Exception_Classifier::CUDA_Error || int( ex.level ) <= 1 )
        can_handle = false;

    if( can_handle )
    {
        int idx_image = -1, idx_chain = -1;

        Log( ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain );
        Log( ex.level, Log_Sender::API,
             fmt::format(
                 "{}:{} in function \'{}\'\n{:>49}Caught exception: {}\n{:>49}Exception backtrace:", file, line,
                 function, " ", message, " " ),
             idx_image, idx_chain );

        // Create backtrace
        Backtrace_Exception();
        Log( ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain );
        Log.Append_to_File();
    }
    else
    {
        auto ex2 = Exception( ex.classifier, ex.level, message, file, line, function );
        std::throw_with_nested( ex2 );
    }
}
catch( ... )
{
    // If something cannot be handled in the core, we re-throw it
    // so it will be handled in the API layer
    spirit_rethrow( message );
}

} // namespace Utility
