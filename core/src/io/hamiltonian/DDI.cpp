#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

void DDI_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Engine::Spin::DDI_Method & ddi_method, intfield & ddi_n_periodic_images, bool & ddi_pb_zero_padding,
    scalar & ddi_radius )
{
    std::string ddi_method_str{};

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // DDI method
        config_file_handle.Read_String( ddi_method_str, "ddi_method" );
        if( ddi_method_str == "none" )
            ddi_method = Engine::Spin::DDI_Method::None;
        else if( ddi_method_str == "fft" )
            ddi_method = Engine::Spin::DDI_Method::FFT;
        else if( ddi_method_str == "fmm" )
            ddi_method = Engine::Spin::DDI_Method::FMM;
        else if( ddi_method_str == "cutoff" )
            ddi_method = Engine::Spin::DDI_Method::Cutoff;
        else
        {
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "Hamiltonian_Heisenberg: Keyword 'ddi_method' got passed invalid method \"{}\". Setting to "
                     "\"none\".",
                     ddi_method_str ) );
            ddi_method_str = "none";
            ddi_method     = Engine::Spin::DDI_Method::None;
        }

        // Number of periodical images
        config_file_handle.Read_3Vector( ddi_n_periodic_images, "ddi_n_periodic_images" );
        config_file_handle.Read_Single( ddi_pb_zero_padding, "ddi_pb_zero_padding" );

        // Dipole-dipole cutoff radius
        config_file_handle.Read_Single( ddi_radius, "ddi_radius" );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read DDI radius from config file \"{}\"", config_file_name ) );
    }

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_method", ddi_method_str ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<21} = ({} {} {})", "ddi_n_periodic_images", ddi_n_periodic_images[0], ddi_n_periodic_images[1],
        ddi_n_periodic_images[2] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_radius", ddi_radius ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_pb_zero_padding", ddi_pb_zero_padding ) );
}

}
