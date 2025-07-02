#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

auto Zeeman_from_TOML( const toml::table & tbl, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::Zeeman::Data
{
    Engine::Spin::Interaction::Zeeman::Data data{ 0, { 0, 0, 1 } };

    scalar magnitude = data.external_field_magnitude;
    read_value( tbl, "external_field_magnitude", magnitude );
    data.external_field_magnitude = magnitude * Utility::Constants::mu_B;

    auto & normal = data.external_field_normal;
    read_value( tbl, "external_field_normal", normal );
    normal.normalize();
    if( normal.norm() < 1e-8 )
    {
        normal = { 0, 0, 1 };
        Log( Log_Level::Warning, Log_Sender::IO,
             "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)" );
    }

    parameter_log.emplace_back( fmt::format( "    {:<21} = {} mu_B", "external field", magnitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "field_normal", normal.transpose() ) );

    return data;
}

auto Zeeman_to_TOML( const Engine::Spin::Interaction::Zeeman::Data * data ) -> toml::table
{
    if( !data )
        return toml::table{};
    else
        return toml::table{
            { "external_field_magnitude", data->external_field_magnitude / Utility::Constants::mu_B },
            { "external_field_normal", toml_array_from_container( data->external_field_normal ) },
        };
}

} // namespace IO
