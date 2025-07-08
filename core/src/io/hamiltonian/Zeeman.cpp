#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

namespace IO
{

auto Zeeman_from_TOML( const toml::table & tbl, std::vector<std::string> & parameter_log )
    -> Engine::Spin::Interaction::Zeeman::Data
{
    Engine::Spin::Interaction::Zeeman::Data data{ 0, { 0, 0, 1 } };

    auto & direction = data.external_field_direction;
    auto & magnitude = data.external_field_magnitude;

    read_Vector3( tbl, "external_field", magnitude, direction );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {} mu_B", "external field", magnitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "field direction", direction.transpose() ) );
    magnitude *= Utility::Constants::mu_B;

    return data;
}

auto Zeeman_to_TOML( const Engine::Spin::Interaction::Zeeman::Data * data ) -> toml::table
{
    if( !data )
        return toml::table{};
    else
        return toml::table{
            { "external_field", as_inline( toml::table{
                                    { "magnitude", data->external_field_magnitude / Utility::Constants::mu_B },
                                    { "direction", toml_array_from_container( data->external_field_direction ) },
                                } ) },
        };
}

} // namespace IO
