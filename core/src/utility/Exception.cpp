#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>

namespace Utility
{
	void rethrow(const std::string & message, const char * file, unsigned int line, const std::string & function)
	{
		try
		{
			std::rethrow_exception(std::current_exception());
		}
		catch( const S_Exception & ex )
		{
			auto ex2 = S_Exception(ex.classifier, ex.level, message, file, line, function);
			std::throw_with_nested(ex2);
		}
		catch( ... )
		{
			auto ex = S_Exception(Exception_Classifier::Unknown_Exception, Log_Level::Error, message, file, line, function);
			std::throw_with_nested(ex);
		}
	}


    void Backtrace_Exception()
    {
        try
        {
            if ( std::exception_ptr eptr = std::current_exception() )
            {
                std::rethrow_exception( eptr );
            }
            else
            {
                Log( Log_Level::Severe, Log_Sender::API, "Unknown Exception. Terminating" );
                Log.Append_to_File();
                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
            }
        }
        catch ( const S_Exception & ex )
        {
            Log( ex.level, Log_Sender::API, std::string(ex.what()));
            try
            {
                rethrow_if_nested(ex);
            }
            catch( ... )
            {
                Backtrace_Exception();
            }
        }
        catch ( const std::exception & ex )
        {
            Log( Log_Level::Severe, Log_Sender::API, std::string(ex.what()));
            try
            {
                rethrow_if_nested(ex);
            }
            catch( ... )
            {
                Backtrace_Exception();
            }
        }
    }

    void Handle_Exception( const std::string & function, int idx_image, int idx_chain )
    {
        try
        {
            // Rethrow in order to get an exception reference (instead of pointer)
            try
            {
                std::rethrow_exception(std::current_exception());
            }
            catch( const S_Exception & ex )
            {
                Log( ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain );
                Log( ex.level, Log_Sender::API, fmt::format("Exception caught in function \'{}\'", function), idx_image, idx_chain );
                if (int(ex.level) > 1)
                    Log( ex.level, Log_Sender::API, "Exception was not severe", idx_image, idx_chain );
                else
                    Log( ex.level, Log_Sender::API, "SEVERE EXCEPTION", idx_image, idx_chain );
                Log( ex.level, Log_Sender::API, "Exception backtrace:", idx_image, idx_chain );

                // Create backtrace
                Backtrace_Exception();
				Log(ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain);
                Log.Append_to_File();
    
                // Check if the exception was recoverable
                if (int(ex.level) <= 1)
                {
                    Log( Log_Level::Severe, Log_Sender::API, "TERMINATING!", idx_image, idx_chain );
					Log.Append_to_File();
                    std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                }
            }
        }
        catch ( ... )
        {
            std::cerr << "Something went super-wrong! TERMINATING!" << std::endl;
            std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
        }
    }


	void spirit_handle_exception_core_func(std::vector<Exception_Classifier> severe_exceptions, std::string message, const char * file, unsigned int line, const std::string & function)
	{
		// Rethrow in order to get an exception reference (instead of pointer)
		try
		{
			std::rethrow_exception(std::current_exception());
		}
		catch (const S_Exception & ex)
		{
			bool can_handle = int(ex.level) > 1 && std::none_of(severe_exceptions.begin(), severe_exceptions.end(), [ex](Exception_Classifier classifier) {return ex.classifier == classifier; });

			if (can_handle)
			{
				int idx_image = -1, idx_chain = -1;

				Log(ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain);
				Log(ex.level, Log_Sender::API, fmt::format("{}:{}\n{:>49}{} \'{}\': {}", file, line, " ", "Exception caught in function", function, message), idx_image, idx_chain);
				Log(ex.level, Log_Sender::API, "Exception backtrace:", idx_image, idx_chain);

				// Create backtrace
				Backtrace_Exception();
				Log(ex.level, Log_Sender::API, "-----------------------------------------------------", idx_image, idx_chain);
				Log.Append_to_File();
			}
			else
			{
				auto ex2 = S_Exception(ex.classifier, ex.level, message, file, line, function);
				std::throw_with_nested(ex2);
			}
		}
		catch (...)
		{
			spirit_rethrow(message);
		}
	}
    
    
    // void Spirit_Exception( const Exception & ex, int idx_image, int idx_chain )
    // {
    //     switch ( ex ) 
    //     {
    //         case Exception::File_not_Found : 
    //             Log( Log_Level::Warning, Log_Sender::API, "File not found. Unable to open.",
    //                  idx_image, idx_chain );
    //             break;
            
    //         case Exception::System_not_Initialized : 
    //             Log( Log_Level::Severe, Log_Sender::API, "System is uninitialized. Terminating.", idx_image, idx_chain );
    //             Log.Append_to_File();
    //             std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
    //             break;
            
    //         case Exception::Division_by_zero:
    //             Log( Log_Level::Severe, Log_Sender::API, "Division by zero", idx_image, idx_chain );
    //             Log.Append_to_File();
    //             std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
    //             break;
            
    //         case Exception::Simulated_domain_too_small:
    //             Log( Log_Level::Error, Log_Sender::API, std::string( "Simulated domain is " ) + 
    //                  std::string( "too small. No action taken." ), idx_image, idx_chain );
    //             break;
            
    //         case Exception::Not_Implemented:
    //             Log( Log_Level::Warning, Log_Sender::API, std::string( "Feature not " ) + 
    //                  std::string( " implemented. No action taken." ), idx_image, idx_chain );
    //             break;
            
    //         case Exception::Non_existing_Image:
    //             Log( Log_Level::Warning, Log_Sender::API, "Non existing image. No action taken.",
    //                  idx_image, idx_chain );
    //             break;
                
    //         case Exception::Non_existing_Chain:
    //             Log( Log_Level::Warning, Log_Sender::API, "Non existing chain. No action taken.",
    //                  idx_image, idx_chain );
    //             break;
    //     }
    // }
}
