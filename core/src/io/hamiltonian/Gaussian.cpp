#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

void Gaussian_from_Config(
    const std::string & config_file_name, std::vector<std::string> & parameter_log, scalarfield & amplitude,
    scalarfield & width, vectorfield & center )
{
    auto n_gaussians = amplitude.size();

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // N
        config_file_handle.Read_Single( n_gaussians, "n_gaussians" );

        // Allocate arrays
        amplitude = scalarfield( n_gaussians, 1.0 );
        width     = scalarfield( n_gaussians, 1.0 );
        center    = vectorfield( n_gaussians, Vector3{ 0, 0, 1 } );
        // Read arrays
        if( config_file_handle.Find( "gaussians" ) )
        {
            for( std::size_t i = 0; i < n_gaussians; ++i )
            {
                config_file_handle.GetLine();
                config_file_handle >> amplitude[i];
                config_file_handle >> width[i];
                for( std::uint8_t j = 0; j < 3; ++j )
                {
                    config_file_handle >> center[i][j];
                }
                center[i].normalize();
            }
        }
        else
            Log( Log_Level::Error, Log_Sender::IO,
                 "Hamiltonian_Gaussian: Keyword 'gaussians' not found. Using Default: 1.0 1.0 {0, 0, 1}" );
    }
    catch( ... )
    {
        spirit_handle_exception_core( fmt::format(
            "Unable to read Hamiltonian_Gaussian parameters from config file  \"{}\"", config_file_name ) );
    }

    parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "n_gaussians", n_gaussians ) );
    if( n_gaussians > 0 )
    {
        parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "amplitude[0]", amplitude[0] ) );
        parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "width[0]", width[0] ) );
        parameter_log.emplace_back( fmt::format( "    {0:<12} = {1}", "center[0]", center[0].transpose() ) );
    }
}

} // namespace IO
