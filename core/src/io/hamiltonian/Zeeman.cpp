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
    auto & magnitude = data.external_field_magnitude;
    auto & normal    = data.external_field_normal;

    read_single( magnitude, tbl["external_field_magnitude"] );
    read_Vector3( normal, tbl["external_field_normal"] );
    normal.normalize();
    if( normal.norm() < 1e-8 )
    {
        normal = { 0, 0, 1 };
        Log( Log_Level::Warning, Log_Sender::IO,
             "Input for 'external_field_normal' had norm zero and has been set to (0,0,1)" );
    }

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "external field", magnitude ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "field_normal", normal.transpose() ) );

    return data;
}

namespace convert
{
namespace Interaction
{
auto Zeeman( const std::string & config_file_name ) -> toml::table
{
    scalar magnitude = 0.0;
    Vector3 normal   = { 0.0, 0.0, 1.0 };

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Read parameters from config if available
        config_file_handle.Read_Single( magnitude, "external_field_magnitude" );
        config_file_handle.Read_Vector3( normal, "external_field_normal" );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read external field from config file \"{}\"", config_file_name ) );
    }

    return toml::table{
        { "external_field_magnitude", magnitude },
        { "external_field_normal", toml_array_from_container( normal ) },
    };
}
} // namespace Interaction

} // namespace convert

void Zeeman_from_Config(
    const std::string & config_file_name, std::vector<std::string> & parameter_log, scalar & magnitude,
    Vector3 & normal )
{
    const auto tbl  = convert::Interaction::Zeeman( config_file_name );
    const auto data = Zeeman_from_TOML( tbl, parameter_log );

    magnitude = data.external_field_magnitude;
    normal    = data.external_field_normal;
}

} // namespace IO
