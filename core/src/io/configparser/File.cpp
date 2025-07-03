#include <io/Configconverter.hpp>
#include <io/Configparser.hpp>

#include <filesystem>
#include <string>

namespace IO
{

auto TOML_from_Config( const std::string & config_file ) -> toml::table
{
    if( std::filesystem::path( config_file ).extension() == ".toml" )
        try
        {
            return toml::parse_file( config_file );
        }
        catch( const std::exception & e )
        {
            spirit_throw(
                Utility::Exception_Classifier::Input_parse_failed, Utility::Log_Level::Error,
                fmt::format( "Error while parsing toml config file \"{}\": {}", config_file, e.what() ) );
            return toml::table{};
        }
    else if( config_file.empty() )
        return toml::table{};
    else
    {
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API,
             "The file \"{}\" is using the deprecated config format. Please convert your config file to toml." );
        return IO::convert::Config( config_file );
    };
}

} // namespace IO
