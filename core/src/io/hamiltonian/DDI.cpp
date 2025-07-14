#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <utility/Enum.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;
namespace Enum = Utility::Enum;

namespace IO
{

auto DDI_from_TOML( const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::DDI::Data
{
    auto ddi_method = Engine::Spin::DDI_Method::None;
    read_enum( tbl, "ddi_method", ddi_method );

    scalar ddi_radius = 0.0;
    read_value( tbl, "ddi_radius", ddi_radius );

    intfield ddi_n_periodic_images{ 4, 4, 4 };
    read_value( tbl, "ddi_n_periodic_images", ddi_n_periodic_images );
    if( ddi_n_periodic_images.size() != 3 )
    {
        Log( Log_Level::Warning, Log_Sender::IO,
             fmt::format(
                 "Wrong size for 'ddi_n_periodic_images', expected 3 got{}, setting to (4, 4, 4)",
                 ddi_n_periodic_images.size() ) );
        ddi_n_periodic_images = { 4, 4, 4 };
    }

    bool ddi_pb_zero_padding = false;
    read_value( tbl, "ddi_pb_zero_padding", ddi_pb_zero_padding );

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_method", Enum::name( ddi_method ) ) );
    parameter_log.emplace_back( fmt::format(
        "    {:<21} = ({} {} {})", "ddi_n_periodic_images", ddi_n_periodic_images[0], ddi_n_periodic_images[1],
        ddi_n_periodic_images[2] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_radius", ddi_radius ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "ddi_pb_zero_padding", ddi_pb_zero_padding ) );

    return { ddi_method, ddi_radius, ddi_pb_zero_padding, ddi_n_periodic_images };
}

auto DDI_to_TOML( const Engine::Spin::Interaction::DDI::Data * data ) -> toml::table
{
    if( !data )
        return toml::table{};

    return toml::table{
        { "ddi_method", Enum::to_string( data->method ) },
        { "ddi_radius", data->cutoff_radius },
        { "pb_zero_padding", data->pb_zero_padding },
        { "n_periodic_images", toml_array_from_container( data->n_periodic_images ) },
    };
}

} // namespace IO
