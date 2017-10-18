#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

namespace Utility
{
    
    void Handle_Exception( int idx_image, int idx_chain )
    {
        try
        {
            try
            {
                if ( std::exception_ptr eptr = std::current_exception() )
                {
                    std::rethrow_exception( eptr );
                }
                else
                {
                    Log( Log_Level::Severe, Log_Sender::API, "Unknown Exception. Terminating", idx_image, idx_chain );
                    Log.Append_to_File();
                    std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                }
            }
            catch ( const std::exception & ex )
            {
                Log( Log_Level::Severe, Log_Sender::API, "Caught std::exception: " + std::string(ex.what()), idx_image, idx_chain );
                Log( Log_Level::Severe, Log_Sender::API, "TERMINATING!", idx_image, idx_chain );
                Log.Append_to_File();

                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
            }
            catch ( const Utility::Exception & ex )
            {
                Spirit_Exception( ex, idx_image, idx_chain );
            }
        }
        catch ( ... )
        {
            std::cerr << "Something went super-wrong! TERMINATING!" << std::endl;
            std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
        }
    }
    
    
    void Spirit_Exception( const Exception & ex, int idx_image, int idx_chain )
    {
        switch ( ex ) 
        {
            case Exception::File_not_Found : 
                Log( Log_Level::Warning, Log_Sender::API, "File not found. Unable to open.",
                     idx_image, idx_chain );
                break;
            
            case Exception::System_not_Initialized : 
                Log( Log_Level::Severe, Log_Sender::API, "System is uninitialized. Terminating.", 
                     idx_image, idx_chain );
                Log.Append_to_File();
                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                break;
            
            case Exception::Division_by_zero:
                Log( Log_Level::Severe, Log_Sender::API, "Division by zero", idx_image, idx_chain );
                Log.Append_to_File();
                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                break;
            
            case Exception::Simulated_domain_too_small:
                Log( Log_Level::Error, Log_Sender::API, std::string( "Simulated domain is too "
                     "small. No action taken." ), idx_image, idx_chain );
                break;
            
            case Exception::Not_Implemented:
                Log( Log_Level::Warning, Log_Sender::API, "Feature not implemented. No action "
                     "taken.", idx_image, idx_chain );
                break;
            
            case Exception::Non_existing_Image:
                Log( Log_Level::Warning, Log_Sender::API, "Non existing image. No action taken.",
                     idx_image, idx_chain );
                break;
                
            case Exception::Non_existing_Chain:
                Log( Log_Level::Warning, Log_Sender::API, "Non existing chain. No action taken.",
                     idx_image, idx_chain );
                break;
                
            case Exception::File_reading_error:
                Log( Log_Level::Error, Log_Sender::IO, "Error while reading file. Operation "
                     "Aborted.", idx_image, idx_chain );
                break;
        }
  
    }

}