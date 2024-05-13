#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

void Zeeman_from_Config(
    const std::string & config_file_name, std::vector<std::string> & parameter_log, scalar & magnitude,
    Vector3 & normal )
{
    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Read parameters from config if available
        config_file_handle.Read_Single( magnitude, "external_field_magnitude" );
        config_file_handle.Read_Vector3( normal, "external_field_normal" );
        normal.normalize();
        if( normal.norm() < 1e-8 )
        {
            normal = { 0, 0, 1 };
            Log( Log_Level::Warning, Log_Sender::IO,
                 "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)" );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read external field from config file \"{}\"", config_file_name ) );
    }

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "external field", magnitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "field_normal", normal.transpose() ) );
}

}
