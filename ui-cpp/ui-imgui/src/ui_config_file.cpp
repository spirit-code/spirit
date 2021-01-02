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

void to_json( Json & j, const BoundingBoxRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
        { "line_width", renderer_widget.line_width },
        { "level_of_detail", renderer_widget.level_of_detail },
        { "draw_shadows", renderer_widget.draw_shadows },
    };
}

void from_json( const Json & j, BoundingBoxRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
    if( j.contains( "line_width" ) )
        j.at( "line_width" ).get_to( renderer_widget.line_width );
    if( j.contains( "level_of_detail" ) )
        j.at( "level_of_detail" ).get_to( renderer_widget.level_of_detail );
    if( j.contains( "draw_shadows" ) )
        j.at( "draw_shadows" ).get_to( renderer_widget.draw_shadows );
}

void to_json( Json & j, const CoordinateSystemRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
    };
}

void from_json( const Json & j, CoordinateSystemRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
}

void to_json( Json & j, const DotRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
        { "size", renderer_widget.size },
    };
}

void from_json( const Json & j, DotRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
    if( j.contains( "size" ) )
        j.at( "size" ).get_to( renderer_widget.size );
}

void to_json( Json & j, const ArrowRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
        { "size", renderer_widget.size },
        { "lod", renderer_widget.lod },
    };
}

void from_json( const Json & j, ArrowRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
    if( j.contains( "size" ) )
        j.at( "size" ).get_to( renderer_widget.size );
    if( j.contains( "lod" ) )
        j.at( "lod" ).get_to( renderer_widget.lod );
}

void to_json( Json & j, const ParallelepipedRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
    };
}

void from_json( const Json & j, ParallelepipedRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
}

void to_json( Json & j, const SphereRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
    };
}

void from_json( const Json & j, SphereRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
}

void to_json( Json & j, const SurfaceRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
    };
}

void from_json( const Json & j, SurfaceRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
}

void to_json( Json & j, const IsosurfaceRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
        { "isovalue", renderer_widget.isovalue },
        { "isocomponent", renderer_widget.isocomponent },
        { "draw_shadows", renderer_widget.draw_shadows },
        { "flip_normals", renderer_widget.flip_normals },
    };
}

void from_json( const Json & j, IsosurfaceRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
    if( j.contains( "isovalue" ) )
        j.at( "isovalue" ).get_to( renderer_widget.isovalue );
    if( j.contains( "isocomponent" ) )
        j.at( "isocomponent" ).get_to( renderer_widget.isocomponent );
    if( j.contains( "draw_shadows" ) )
        j.at( "draw_shadows" ).get_to( renderer_widget.draw_shadows );
    if( j.contains( "flip_normals" ) )
        j.at( "flip_normals" ).get_to( renderer_widget.flip_normals );
}

void to_json( Json & j, const std::shared_ptr<RendererWidget> & renderer_widget )
{
    if( dynamic_cast<BoundingBoxRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "BoundingBoxRendererWidget",
                    *dynamic_cast<BoundingBoxRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<CoordinateSystemRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "CoordinateSystemRendererWidget",
                    *dynamic_cast<CoordinateSystemRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<DotRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "DotRendererWidget", *dynamic_cast<DotRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<ArrowRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "ArrowRendererWidget", *dynamic_cast<ArrowRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<ParallelepipedRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "ParallelepipedRendererWidget",
                    *dynamic_cast<ParallelepipedRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<SphereRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "SphereRendererWidget", *dynamic_cast<SphereRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<SurfaceRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "SurfaceRendererWidget", *dynamic_cast<SurfaceRendererWidget *>( renderer_widget.get() ) } };
    }
    else if( dynamic_cast<IsosurfaceRendererWidget *>( renderer_widget.get() ) )
    {
        j = Json{ { "IsosurfaceRendererWidget", *dynamic_cast<IsosurfaceRendererWidget *>( renderer_widget.get() ) } };
    }
}

// -----------------------------------------------------------------------------------

UiConfigFile::UiConfigFile( UiSharedState & ui_shared_state, RenderingLayer & rendering_layer )
        : ui_shared_state( ui_shared_state ), rendering_layer( rendering_layer )
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
            if( group.contains( "parameters_widget" ) )
                group.at( "parameters_widget" ).get_to( this->show_parameters_widget );
            if( group.contains( "configurations_widget" ) )
                group.at( "configurations_widget" ).get_to( this->show_configurations_widget );
            if( group.contains( "visualisation_widget" ) )
                group.at( "visualisation_widget" ).get_to( this->show_visualisation_widget );
            if( group.contains( "hamiltonian_widget" ) )
                group.at( "hamiltonian_widget" ).get_to( this->show_hamiltonian_widget );
            if( group.contains( "geometry_widget" ) )
                group.at( "geometry_widget" ).get_to( this->show_geometry_widget );
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
            auto group = settings_json.at( "shared_state" );
            if( group.contains( "dark_mode" ) )
                group.at( "dark_mode" ).get_to( this->ui_shared_state.dark_mode );
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

        if( settings_json.contains( "visualisation" ) )
        {
            auto & group = settings_json.at( "visualisation" );
            if( group.contains( "renderers" ) )
            {
                auto & renderers = group.at( "renderers" );
                for( auto & j : renderers )
                {
                    if( j.contains( "BoundingBoxRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<BoundingBoxRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "BoundingBoxRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "CoordinateSystemRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<CoordinateSystemRendererWidget>( rendering_layer.state );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "CoordinateSystemRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "DotRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<DotRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "DotRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "ArrowRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<ArrowRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "ArrowRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "ParallelepipedRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<ParallelepipedRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "ParallelepipedRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "SphereRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<SphereRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "SphereRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "SurfaceRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<SurfaceRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "SurfaceRendererWidget" ).get_to( *ptr );
                    }
                    if( j.contains( "IsosurfaceRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<IsosurfaceRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield );
                        rendering_layer.renderer_widgets.push_back( ptr );
                        j.at( "IsosurfaceRendererWidget" ).get_to( *ptr );
                    }
                }
            }
            if( group.contains( "camera_position" ) )
            {
                auto camera_position = rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>();
                camera_position.x    = float( group.at( "camera_position" )[0] );
                camera_position.y    = float( group.at( "camera_position" )[1] );
                camera_position.z    = float( group.at( "camera_position" )[2] );
                rendering_layer.view.setOption<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
            }
            if( group.contains( "center_position" ) )
            {
                auto center_position = rendering_layer.view.options().get<VFRendering::View::Option::CENTER_POSITION>();
                center_position.x    = group.at( "center_position" )[0];
                center_position.y    = group.at( "center_position" )[1];
                center_position.z    = group.at( "center_position" )[2];
                rendering_layer.view.setOption<VFRendering::View::Option::CENTER_POSITION>( center_position );
            }
            if( group.contains( "up_vector" ) )
            {
                auto up_vector = rendering_layer.view.options().get<VFRendering::View::Option::UP_VECTOR>();
                up_vector.x    = group.at( "up_vector" )[0];
                up_vector.y    = group.at( "up_vector" )[1];
                up_vector.z    = group.at( "up_vector" )[2];
                rendering_layer.view.setOption<VFRendering::View::Option::UP_VECTOR>( up_vector );
            }
        }
    }
}

void UiConfigFile::to_json() const
{
    auto camera_pos = rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>();
    auto center_pos = rendering_layer.view.options().get<VFRendering::View::Option::CENTER_POSITION>();
    auto up_vector  = rendering_layer.view.options().get<VFRendering::View::Option::UP_VECTOR>();

    Json settings_json = {
        {
            "show",
            {
                { "parameters_widget", this->show_parameters_widget },
                { "configurations_widget", this->show_configurations_widget },
                { "visualisation_widget", this->show_visualisation_widget },
                { "hamiltonian_widget", this->show_hamiltonian_widget },
                { "geometry_widget", this->show_geometry_widget },
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

        {
            "visualisation",
            {
                { "camera_position", { camera_pos.x, camera_pos.y, camera_pos.z } },
                { "center_position", { center_pos.x, center_pos.y, center_pos.z } },
                { "up_vector", { up_vector.x, up_vector.y, up_vector.z } },
                { "renderers", this->rendering_layer.renderer_widgets },
            },
        },
    };

    auto file_path = fs::path( this->settings_filename );
    std::ofstream ofs( file_path );
    ofs << std::setw( 2 ) << settings_json;
    ofs.close();
}

} // namespace ui