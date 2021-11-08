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

void to_json( Json & j, const RendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
        { "filter_direction_min", renderer_widget.filter_direction_min },
        { "filter_direction_max", renderer_widget.filter_direction_max },
        { "filter_position_min", renderer_widget.filter_position_min },
        { "filter_position_max", renderer_widget.filter_position_max },
    };
}

void from_json( const Json & j, RendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
    if( j.contains( "filter_direction_min" ) )
        j.at( "filter_direction_min" ).get_to( renderer_widget.filter_direction_min );
    if( j.contains( "filter_direction_max" ) )
        j.at( "filter_direction_max" ).get_to( renderer_widget.filter_direction_max );
    if( j.contains( "filter_position_min" ) )
        j.at( "filter_position_min" ).get_to( renderer_widget.filter_position_min );
    if( j.contains( "filter_position_max" ) )
        j.at( "filter_position_max" ).get_to( renderer_widget.filter_position_max );
}

void to_json( Json & j, const BoundingBoxRendererWidget & renderer_widget )
{
    j = Json{
        { "show", renderer_widget.show_ },
        { "colour_dark", renderer_widget.colour_dark },
        { "colour_light", renderer_widget.colour_light },
        { "line_width", renderer_widget.line_width },
        { "level_of_detail", renderer_widget.level_of_detail },
        { "draw_shadows", renderer_widget.draw_shadows },
    };
}

void from_json( const Json & j, BoundingBoxRendererWidget & renderer_widget )
{
    if( j.contains( "show" ) )
        j.at( "show" ).get_to( renderer_widget.show_ );
    if( j.contains( "colour_dark" ) )
        j.at( "colour_dark" ).get_to( renderer_widget.colour_dark );
    if( j.contains( "colour_light" ) )
        j.at( "colour_light" ).get_to( renderer_widget.colour_light );
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

void to_json( Json & j, const ColormapWidget & renderer_widget )
{
    j = Json{
        { "colormap", renderer_widget.colormap },
        { "colormap_monochrome_color",
          { renderer_widget.colormap_monochrome_color.x, renderer_widget.colormap_monochrome_color.y,
            renderer_widget.colormap_monochrome_color.z } },
        { "colormap_cardinal_a",
          { renderer_widget.colormap_cardinal_a.x, renderer_widget.colormap_cardinal_a.y,
            renderer_widget.colormap_cardinal_a.z } },
        { "colormap_cardinal_b",
          { renderer_widget.colormap_cardinal_b.x, renderer_widget.colormap_cardinal_b.y,
            renderer_widget.colormap_cardinal_b.z } },
        { "colormap_cardinal_c",
          { renderer_widget.colormap_cardinal_c.x, renderer_widget.colormap_cardinal_c.y,
            renderer_widget.colormap_cardinal_c.z } },
        { "colormap_rotation", renderer_widget.colormap_rotation },
        { "colormap_invert_z", renderer_widget.colormap_invert_z },
        { "colormap_invert_xy", renderer_widget.colormap_invert_xy },
    };
}

void from_json( const Json & j, ColormapWidget & renderer_widget )
{
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( renderer_widget.colormap );
    if( j.contains( "colormap_monochrome_color" ) )
    {
        renderer_widget.colormap_monochrome_color.x = j.at( "colormap_monochrome_color" )[0];
        renderer_widget.colormap_monochrome_color.y = j.at( "colormap_monochrome_color" )[1];
        renderer_widget.colormap_monochrome_color.z = j.at( "colormap_monochrome_color" )[2];
    }
    if( j.contains( "colormap_cardinal_a" ) )
    {
        renderer_widget.colormap_cardinal_a.x = j.at( "colormap_cardinal_a" )[0];
        renderer_widget.colormap_cardinal_a.y = j.at( "colormap_cardinal_a" )[1];
        renderer_widget.colormap_cardinal_a.z = j.at( "colormap_cardinal_a" )[2];
    }
    if( j.contains( "colormap_cardinal_b" ) )
    {
        renderer_widget.colormap_cardinal_b.x = j.at( "colormap_cardinal_b" )[0];
        renderer_widget.colormap_cardinal_b.y = j.at( "colormap_cardinal_b" )[1];
        renderer_widget.colormap_cardinal_b.z = j.at( "colormap_cardinal_b" )[2];
    }
    if( j.contains( "colormap_cardinal_c" ) )
    {
        renderer_widget.colormap_cardinal_c.x = j.at( "colormap_cardinal_c" )[0];
        renderer_widget.colormap_cardinal_c.y = j.at( "colormap_cardinal_c" )[1];
        renderer_widget.colormap_cardinal_c.z = j.at( "colormap_cardinal_c" )[2];
    }
    if( j.contains( "colormap_rotation" ) )
        j.at( "colormap_rotation" ).get_to( renderer_widget.colormap_rotation );
    if( j.contains( "colormap_invert_z" ) )
        j.at( "colormap_invert_z" ).get_to( renderer_widget.colormap_invert_z );
    if( j.contains( "colormap_invert_xy" ) )
        j.at( "colormap_invert_xy" ).get_to( renderer_widget.colormap_invert_xy );
}

void to_json( Json & j, const DotRendererWidget & renderer_widget )
{
    j = Json{
        { "renderer", static_cast<const RendererWidget &>( renderer_widget ) },
        { "colormap", static_cast<const ColormapWidget &>( renderer_widget ) },
        { "size", renderer_widget.size },
    };
}

void from_json( const Json & j, DotRendererWidget & renderer_widget )
{
    if( j.contains( "renderer" ) )
        j.at( "renderer" ).get_to( static_cast<RendererWidget &>( renderer_widget ) );
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( static_cast<ColormapWidget &>( renderer_widget ) );
    if( j.contains( "size" ) )
        j.at( "size" ).get_to( renderer_widget.size );
}

void to_json( Json & j, const ArrowRendererWidget & renderer_widget )
{
    j = Json{
        { "renderer", static_cast<const RendererWidget &>( renderer_widget ) },
        { "colormap", static_cast<const ColormapWidget &>( renderer_widget ) },
        { "size", renderer_widget.size },
        { "lod", renderer_widget.lod },
    };
}

void from_json( const Json & j, ArrowRendererWidget & renderer_widget )
{
    if( j.contains( "renderer" ) )
        j.at( "renderer" ).get_to( static_cast<RendererWidget &>( renderer_widget ) );
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( static_cast<ColormapWidget &>( renderer_widget ) );
    if( j.contains( "size" ) )
        j.at( "size" ).get_to( renderer_widget.size );
    if( j.contains( "lod" ) )
        j.at( "lod" ).get_to( renderer_widget.lod );
}

void to_json( Json & j, const ParallelepipedRendererWidget & renderer_widget )
{
    j = Json{
        { "renderer", static_cast<const RendererWidget &>( renderer_widget ) },
        { "colormap", static_cast<const ColormapWidget &>( renderer_widget ) },
        { "size", renderer_widget.size },
    };
}

void from_json( const Json & j, ParallelepipedRendererWidget & renderer_widget )
{
    if( j.contains( "renderer" ) )
        j.at( "renderer" ).get_to( static_cast<RendererWidget &>( renderer_widget ) );
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( static_cast<ColormapWidget &>( renderer_widget ) );
    if( j.contains( "size" ) )
        j.at( "size" ).get_to( renderer_widget.size );
}

void to_json( Json & j, const SphereRendererWidget & renderer_widget )
{
    j = Json{
        { "renderer", static_cast<const RendererWidget &>( renderer_widget ) },
        { "colormap", static_cast<const ColormapWidget &>( renderer_widget ) },
        { "size", renderer_widget.size },
        { "lod", renderer_widget.lod },
    };
}

void from_json( const Json & j, SphereRendererWidget & renderer_widget )
{
    if( j.contains( "renderer" ) )
        j.at( "renderer" ).get_to( static_cast<RendererWidget &>( renderer_widget ) );
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( static_cast<ColormapWidget &>( renderer_widget ) );
    if( j.contains( "size" ) )
        j.at( "size" ).get_to( renderer_widget.size );
    if( j.contains( "lod" ) )
        j.at( "lod" ).get_to( renderer_widget.lod );
}

void to_json( Json & j, const SurfaceRendererWidget & renderer_widget )
{
    j = Json{
        { "renderer", static_cast<const RendererWidget &>( renderer_widget ) },
        { "colormap", static_cast<const ColormapWidget &>( renderer_widget ) },
    };
}

void from_json( const Json & j, SurfaceRendererWidget & renderer_widget )
{
    if( j.contains( "renderer" ) )
        j.at( "renderer" ).get_to( static_cast<RendererWidget &>( renderer_widget ) );
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( static_cast<ColormapWidget &>( renderer_widget ) );
}

void to_json( Json & j, const IsosurfaceRendererWidget & renderer_widget )
{
    j = Json{
        { "renderer", static_cast<const RendererWidget &>( renderer_widget ) },
        { "colormap", static_cast<const ColormapWidget &>( renderer_widget ) },
        { "isovalue", renderer_widget.isovalue },
        { "isocomponent", renderer_widget.isocomponent },
        { "draw_shadows", renderer_widget.draw_shadows },
        { "flip_normals", renderer_widget.flip_normals },
    };
}

void from_json( const Json & j, IsosurfaceRendererWidget & renderer_widget )
{
    if( j.contains( "renderer" ) )
        j.at( "renderer" ).get_to( static_cast<RendererWidget &>( renderer_widget ) );
    if( j.contains( "colormap" ) )
        j.at( "colormap" ).get_to( static_cast<ColormapWidget &>( renderer_widget ) );
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
            if( group.contains( "interaction_radius" ) )
                group.at( "interaction_radius" ).get_to( this->interaction_radius );
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
            if( group.contains( "min_apply_to_all" ) )
                group.at( "min_apply_to_all" ).get_to( this->ui_shared_state.min_apply_to_all );
            if( group.contains( "mc_apply_to_all" ) )
                group.at( "mc_apply_to_all" ).get_to( this->ui_shared_state.mc_apply_to_all );
            if( group.contains( "llg_apply_to_all" ) )
                group.at( "llg_apply_to_all" ).get_to( this->ui_shared_state.llg_apply_to_all );
            if( group.contains( "mmf_apply_to_all" ) )
                group.at( "mmf_apply_to_all" ).get_to( this->ui_shared_state.mmf_apply_to_all );
            if( group.contains( "ema_apply_to_all" ) )
                group.at( "ema_apply_to_all" ).get_to( this->ui_shared_state.ema_apply_to_all );
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

        auto update = [&]( std::shared_ptr<RendererWidget> ptr )
        {
            ptr->apply_settings();
            rendering_layer.renderer_widgets.push_back( ptr );
            ptr->id = rendering_layer.renderer_id_counter;
            ++rendering_layer.renderer_id_counter;
        };

        if( settings_json.contains( "visualisation" ) )
        {
            auto & group = settings_json.at( "visualisation" );
            if( group.contains( "boundingbox_renderer" ) )
            {
                auto ptr = std::make_shared<BoundingBoxRendererWidget>(
                    rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                    rendering_layer.ui_shared_state, rendering_layer.vfr_update_deque );
                rendering_layer.boundingbox_renderer_widget = ptr;
                group.at( "boundingbox_renderer" ).get_to( *ptr );
                ptr->apply_settings();
            }
            if( group.contains( "coordinatesystem_renderer" ) )
            {
                auto ptr = std::make_shared<CoordinateSystemRendererWidget>(
                    rendering_layer.state, rendering_layer.view, rendering_layer.vfr_update_deque );
                rendering_layer.coordinatesystem_renderer_widget = ptr;
                group.at( "coordinatesystem_renderer" ).get_to( *ptr );
                ptr->apply_settings();
            }
            if( group.contains( "renderers" ) )
            {
                auto & renderers = group.at( "renderers" );
                for( auto & j : renderers )
                {
                    if( j.contains( "DotRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<DotRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                            rendering_layer.vfr_update_deque );
                        j.at( "DotRendererWidget" ).get_to( *ptr );
                        update( ptr );
                    }
                    if( j.contains( "ArrowRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<ArrowRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                            rendering_layer.vfr_update_deque );
                        j.at( "ArrowRendererWidget" ).get_to( *ptr );
                        update( ptr );
                    }
                    if( j.contains( "ParallelepipedRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<ParallelepipedRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                            rendering_layer.vfr_update_deque );
                        j.at( "ParallelepipedRendererWidget" ).get_to( *ptr );
                        update( ptr );
                    }
                    if( j.contains( "SphereRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<SphereRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                            rendering_layer.vfr_update_deque );
                        j.at( "SphereRendererWidget" ).get_to( *ptr );
                        update( ptr );
                    }
                    if( j.contains( "SurfaceRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<SurfaceRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                            rendering_layer.vfr_update_deque );
                        j.at( "SurfaceRendererWidget" ).get_to( *ptr );
                        update( ptr );
                    }
                    if( j.contains( "IsosurfaceRendererWidget" ) )
                    {
                        auto ptr = std::make_shared<IsosurfaceRendererWidget>(
                            rendering_layer.state, rendering_layer.view, rendering_layer.vectorfield,
                            rendering_layer.vfr_update_deque );
                        j.at( "IsosurfaceRendererWidget" ).get_to( *ptr );
                        update( ptr );
                    }
                }
            }
            if( group.contains( "camera_position" ) )
            {
                auto camera_position = rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>();
                camera_position.x    = float( group.at( "camera_position" )[0] );
                camera_position.y    = float( group.at( "camera_position" )[1] );
                camera_position.z    = float( group.at( "camera_position" )[2] );
                rendering_layer.set_view_option<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
            }
            if( group.contains( "center_position" ) )
            {
                auto center_position = rendering_layer.view.options().get<VFRendering::View::Option::CENTER_POSITION>();
                center_position.x    = group.at( "center_position" )[0];
                center_position.y    = group.at( "center_position" )[1];
                center_position.z    = group.at( "center_position" )[2];
                rendering_layer.set_view_option<VFRendering::View::Option::CENTER_POSITION>( center_position );
            }
            if( group.contains( "up_vector" ) )
            {
                auto up_vector = rendering_layer.view.options().get<VFRendering::View::Option::UP_VECTOR>();
                up_vector.x    = group.at( "up_vector" )[0];
                up_vector.y    = group.at( "up_vector" )[1];
                up_vector.z    = group.at( "up_vector" )[2];
                rendering_layer.set_view_option<VFRendering::View::Option::UP_VECTOR>( up_vector );
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
                { "interaction_radius", this->interaction_radius },
            },
        },

        {
            "shared_state",
            {
                { "dark_mode", this->ui_shared_state.dark_mode },
                { "background_dark", this->ui_shared_state.background_dark },
                { "background_light", this->ui_shared_state.background_light },
                { "light_direction", this->ui_shared_state.light_direction },
                { "min_apply_to_all", this->ui_shared_state.min_apply_to_all },
                { "mc_apply_to_all", this->ui_shared_state.mc_apply_to_all },
                { "llg_apply_to_all", this->ui_shared_state.llg_apply_to_all },
                { "mmf_apply_to_all", this->ui_shared_state.mmf_apply_to_all },
                { "ema_apply_to_all", this->ui_shared_state.ema_apply_to_all },
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
                { "boundingbox_renderer", *this->rendering_layer.boundingbox_renderer_widget },
                { "coordinatesystem_renderer", *this->rendering_layer.coordinatesystem_renderer_widget },
            },
        },
    };

    auto file_path = fs::path( this->settings_filename );
    std::ofstream ofs( file_path );
    ofs << std::setw( 2 ) << settings_json;
    ofs.close();
}

} // namespace ui