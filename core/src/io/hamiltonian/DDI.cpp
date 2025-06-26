#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace convert
{

namespace Interaction
{

auto DDI( const std::string & config_file_name ) -> toml::table
{
    std::string ddi_method_str{};
    intfield ddi_n_periodic_images = { 4, 4, 4 };
    bool ddi_pb_zero_padding       = false;
    scalar ddi_radius              = 0.0;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        config_file_handle.Read_String( ddi_method_str, "ddi_method" );
        config_file_handle.Read_3Vector( ddi_n_periodic_images, "ddi_n_periodic_images" );
        config_file_handle.Read_Single( ddi_pb_zero_padding, "ddi_pb_zero_padding" );
        config_file_handle.Read_Single( ddi_radius, "ddi_radius" );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read DDI radius from config file \"{}\"", config_file_name ) );
    };

    if( ddi_method_str.empty() || ddi_radius == 0 )
        return toml::table{};
    else
        return toml::table{
            { "ddi_method", ddi_method_str },
            { "ddi_radius", ddi_radius },
            { "ddi_n_periodic_images", toml_array_from_container( ddi_n_periodic_images ) },
            { "ddi_pb_zero_padding", ddi_pb_zero_padding },
        };
}

} // namespace Interaction

} // namespace convert

void DDI_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Engine::Spin::DDI_Method & ddi_method, intfield & ddi_n_periodic_images, bool & ddi_pb_zero_padding,
    scalar & ddi_radius )
{
    const auto [ddi_method_str, ddi_method_] = [&tbl]() -> std::pair<std::string_view, Engine::Spin::DDI_Method>
    {
        auto method_str = tbl["ddi_method"].value<std::string>();
        if( !method_str )
            return { "none", Engine::Spin::DDI_Method::None };

        if( *method_str == "none" )
            return { "none", Engine::Spin::DDI_Method::None };
        else if( *method_str == "fft" )
            return { "fft", Engine::Spin::DDI_Method::FFT };
        else if( *method_str == "fmm" )
            return { "ffm", Engine::Spin::DDI_Method::FMM };
        else if( *method_str == "cutoff" )
            return { "cutoff", Engine::Spin::DDI_Method::Cutoff };
        else
        {
            if( *method_str != "none" )
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Keyword 'ddi_method' got passed invalid method \"{}\". Setting to \"none\".", *method_str ) );
            return { "none", Engine::Spin::DDI_Method::None };
        }
    }();
    ddi_method = ddi_method_;
    ddi_radius = tbl["ddi_radius"].value_or<scalar>( 0.0 );

    ddi_n_periodic_images = [&tbl]
    {
        auto result = intfield{ 4, 4, 4 };
        auto array  = tbl["ddi_n_periodic_images"];
        if( !array )
            return result;

        try
        {
            result = toml_array_transform<intfield>::transform( *array.node() );
            if( result.size() != 3 )
            {
                Log( Log_Level::Warning, Log_Sender::IO,
                     "Wrong sized array 'ddi_n_periodic_images', expected 3, found {}. Setting to (4,4,4)..." );
                result = { 4, 4, 4 };
            }
        }
        catch( ... )
        {
            spirit_handle_exception_core( "Failed reading 'ddi_n_periodic_images'" );
        }

        return result;
    }();

    ddi_pb_zero_padding = tbl["ddi_pb_zero_padding"].value_or( false );

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_method", ddi_method_str ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<21} = ({} {} {})", "ddi_n_periodic_images", ddi_n_periodic_images[0], ddi_n_periodic_images[1],
        ddi_n_periodic_images[2] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_radius", ddi_radius ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_pb_zero_padding", ddi_pb_zero_padding ) );
}

void DDI_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    Engine::Spin::DDI_Method & ddi_method, intfield & ddi_n_periodic_images, bool & ddi_pb_zero_padding,
    scalar & ddi_radius )
{
    return DDI_from_TOML(
        convert::Interaction::DDI( config_file_name ), geometry, parameter_log, ddi_method, ddi_n_periodic_images,
        ddi_pb_zero_padding, ddi_radius );
}

} // namespace IO
