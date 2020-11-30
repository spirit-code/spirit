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

        if( settings_json.contains( "show" ) )
        {
            auto & group = settings_json.at( "show" );
            if( group.contains( "parameters_settings" ) )
                group.at( "parameters_settings" ).get_to( this->show_parameters_settings );
            if( group.contains( "visualisation_settings" ) )
                group.at( "visualisation_settings" ).get_to( this->show_visualisation_settings );
            if( group.contains( "hamiltonian_settings" ) )
                group.at( "hamiltonian_settings" ).get_to( this->show_hamiltonian_settings );
            if( group.contains( "geometry_settings" ) )
                group.at( "geometry_settings" ).get_to( this->show_geometry_settings );
            if( group.contains( "plots" ) )
                group.at( "plots" ).get_to( this->show_plots );
            if( group.contains( "settings" ) )
                group.at( "settings" ).get_to( this->show_settings );
        }

        if( settings_json.contains( "overlays" ) )
        {
            auto & group = settings_json.at( "overlays" );
            if( group.contains( "show" ) )
                group.at( "show" ).get_to( this->show_overlays );
            if( group.contains( "system_corner" ) )
                group.at( "system_corner" ).get_to( this->overlay_system_corner );
            if( group.contains( "system_position" ) )
                group.at( "system_position" ).get_to( this->overlay_system_position );
            if( group.contains( "calculation_corner" ) )
                group.at( "calculation_corner" ).get_to( this->overlay_calculation_corner );
            if( group.contains( "calculation_position" ) )
                group.at( "calculation_position" ).get_to( this->overlay_calculation_position );
        }

        if( settings_json.contains( "shared_state" ) )
        {
            auto & group = settings_json.at( "shared_state" );
            if( settings_json.contains( "dark_mode" ) )
                settings_json.at( "dark_mode" ).get_to( this->ui_shared_state.dark_mode );
            if( group.contains( "background_dark" ) )
                group.at( "background_dark" ).get_to( this->ui_shared_state.background_dark );
            if( group.contains( "background_light" ) )
                group.at( "background_light" ).get_to( this->ui_shared_state.background_light );
            if( group.contains( "light_direction" ) )
                group.at( "light_direction" ).get_to( this->ui_shared_state.light_direction );
        }

        if( settings_json.contains( "main_window" ) )
        {
            auto & group = settings_json.at( "main_window" );
            if( group.contains( "size" ) )
                group.at( "size" ).get_to( this->window_size );
            if( group.contains( "pos" ) )
                group.at( "pos" ).get_to( this->window_position );
            if( group.contains( "maximized" ) )
                group.at( "maximized" ).get_to( this->window_maximized );
        }
    }
}

void UiConfigFile::to_json() const
{
    Json settings_json = {
        {
            "show",
            {
                { "parameters_settings", this->show_parameters_settings },
                { "visualisation_settings", this->show_visualisation_settings },
                { "hamiltonian_settings", this->show_hamiltonian_settings },
                { "geometry_settings", this->show_geometry_settings },
                { "plots", this->show_plots },
                { "settings", this->show_settings },
            },
        },

        {
            "overlays",
            {
                { "show", this->show_overlays },
                { "system_corner", this->overlay_system_corner },
                { "system_position", this->overlay_system_position },
                { "calculation_corner", this->overlay_calculation_corner },
                { "calculation_position", this->overlay_calculation_position },

            },
        },

        {
            "shared_state",
            {
                { "dark_mode", this->ui_shared_state.dark_mode },
                { "background_dark", this->ui_shared_state.background_dark },
                { "background_light", this->ui_shared_state.background_light },
                { "light_direction", this->ui_shared_state.light_direction },
            },
        },

        {
            "main_window",
            {
                { "position", { this->window_position[0], this->window_position[1] } },
                { "size", { this->window_size[0], this->window_size[1] } },
                { "maximized", this->window_maximized },
            },
        },
    };

    auto file_path = fs::path( this->settings_filename );
    std::ofstream ofs( file_path );
    ofs << std::setw( 2 ) << settings_json;
    ofs.close();
}

} // namespace ui