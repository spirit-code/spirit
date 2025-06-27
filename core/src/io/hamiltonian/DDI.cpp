#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <io/configparser/Converter.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

auto DDI_from_TOML( const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::DDI::Data
{
    const auto [ddi_method_str, ddi_method] = [&tbl]() -> std::pair<std::string_view, Engine::Spin::DDI_Method>
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

    const auto ddi_radius = tbl["ddi_radius"].value_or<scalar>( 0.0 );

    const auto ddi_n_periodic_images = [&tbl]
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

    const auto ddi_pb_zero_padding = tbl["ddi_pb_zero_padding"].value_or( false );

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_method", ddi_method_str ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<21} = ({} {} {})", "ddi_n_periodic_images", ddi_n_periodic_images[0], ddi_n_periodic_images[1],
        ddi_n_periodic_images[2] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_radius", ddi_radius ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_pb_zero_padding", ddi_pb_zero_padding ) );

    return { ddi_method, ddi_radius, ddi_pb_zero_padding, ddi_n_periodic_images };
}

} // namespace IO
