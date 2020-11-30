#include <ui_config_file.hpp>

#include <nlohmann/json.hpp>

#include <fmt/format.h>

#include <fstream>
#include <iomanip>

#if defined( __cplusplus ) && __cplusplus >= 201703L && defined( __has_include )
#if __has_include( <filesystem>)
#define GHC_USE_STD_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif
#endif
#ifndef GHC_USE_STD_FS
#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;
#endif

using Json = nlohmann::json;

namespace ui
{

UiConfigFile::UiConfigFile( UiSharedState & ui_shared_state ) : ui_shared_state( ui_shared_state )
{
    this->from_json();
}

UiConfigFile::~UiConfigFile()
{
    this->to_json();
}

void UiConfigFile::from_json()
{
    auto file_path = fs::path( this->settings_filename );
    if( fs::is_regular_file( file_path ) )
    {
        Json settings_json;
        std::ifstream ifs( file_path );
        ifs >> settings_json;
        ifs.close();

        if( settings_json.contains( "dark_mode" ) )
            settings_json.at( "dark_mode" ).get_to( this->ui_shared_state.dark_mode );
        if( settings_json.contains( "visualisation" ) )
        {
            if( settings_json.at( "visualisation" ).contains( "background_dark" ) )
                settings_json.at( "visualisation" )
                    .at( "background_dark" )
                    .get_to( this->ui_shared_state.background_dark );
            if( settings_json.at( "visualisation" ).contains( "background_light" ) )
                settings_json.at( "visualisation" )
                    .at( "background_light" )
                    .get_to( this->ui_shared_state.background_light );
            if( settings_json.at( "visualisation" ).contains( "light_direction" ) )
                settings_json.at( "visualisation" )
                    .at( "light_direction" )
                    .get_to( this->ui_shared_state.light_direction );
        }

        if( settings_json.contains( "main_window" ) )
        {
            if( settings_json.at( "main_window" ).contains( "size" ) )
                settings_json.at( "main_window" ).at( "size" ).get_to( this->window_size );
            if( settings_json.at( "main_window" ).contains( "pos" ) )
                settings_json.at( "main_window" ).at( "pos" ).get_to( this->window_position );
        }
    }
}

void UiConfigFile::to_json() const
{
    Json settings_json = {
        { "dark_mode", this->ui_shared_state.dark_mode },
        { "visualisation",
          { { "background_dark", this->ui_shared_state.background_dark },
            { "background_light", this->ui_shared_state.background_light },
            { "light_direction", this->ui_shared_state.light_direction } } },
        { "main_window",
          { { "position", { this->window_position[0], this->window_position[1] } },
            { "size", { this->window_size[0], this->window_size[1] } },
            { "collapsed", false } } },
    };

    auto file_path = fs::path( this->settings_filename );
    std::ofstream ofs( file_path );
    ofs << std::setw( 2 ) << settings_json;
    ofs.close();
}

} // namespace ui