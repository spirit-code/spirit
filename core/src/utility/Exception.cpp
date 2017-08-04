#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

namespace Utility
{
    void Handle_exception( const Exception & ex, int idx_image=-1, int idx_chain=-1 )
    {
        switch ( ex ) 
        {
            case Exception::File_not_Found : 
                Log( Log_Level::Warning, Log_Sender::API, "File not found. Unable to open ",
                     idx_image, idx_chain );
                break;
            
            case Exception::System_not_Initialized : 
                Log( Log_Level::Error, Log_Sender::API, "System is uninitialized. Terminating",
                     idx_image, idx_chain );
                Log.Append_to_File();
                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                break;
            
            case Exception::Division_by_zero:
                Log( Log_Level::Error, Log_Sender::API, "Division by zero", idx_image, idx_chain );
                Log.Append_to_File();
                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                break;
            
            // XXX: should the next two cases be Error or Warning?
            
            case Exception::Simulated_domain_too_small:
                Log( Log_Level::Warning, Log_Sender::API, std::string( "Simulated domain is " ) + 
                     std::string( "too small. No action taken." ), idx_image, idx_chain );
                break;
            
            case Exception::Not_Implemented:
                Log( Log_Level::Warning, Log_Sender::API, std::string( "Feature not " ) + 
                     std::string( " implemented. No action taken." ), idx_image, idx_chain );
                break;
            
            case Exception::Unknown_Exception:
                Log( Log_Level::Error, Log_Sender::API, " Unknown Exception. Terminating",
                     idx_image, idx_chain );
                Log.Append_to_File();
                std::exit( EXIT_FAILURE );  // exit the application. may lead to data loss
                break;
            
            case Exception::Non_existing_Image:
                Log( Log_Level::Warning, Log_Sender::API, "Non existing image. No action taken.",
                     idx_image, idx_chain );
                break;
                
            case Exception::Non_existing_Chain:
                Log( Log_Level::Warning, Log_Sender::API, "Non existing chain. No action taken.",
                     idx_image, idx_chain );
                break;
        }
  
    }

}