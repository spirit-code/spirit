#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
// #include <GLES3/gl3.h>
#endif

#include <glad/glad.h>

#include <enums.hpp>
#include <fonts.hpp>
#include <images.hpp>
#include <main_window.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <imgui/imgui_internal.h>

#include <implot/implot.h>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>

#include <fmt/format.h>

#include <nfd.h>

#include <cmath>
#include <exception>
#include <map>
#include <string>

static ui::MainWindow * global_window_handle;
static bool fullscreen_toggled = false;
#ifdef SPIRIT_ENABLE_PINNING
static const bool SPIRIT_PINNING_ENABLED = true;
#else
static const bool SPIRIT_PINNING_ENABLED = false;
#endif

/////////////////////////////////////////////////////////////////////

#ifdef __EMSCRIPTEN__
EM_JS( int, canvas_get_width, (), { return Module.canvas.width; } );
EM_JS( int, canvas_get_height, (), { return Module.canvas.height; } );
EM_JS( void, resizeCanvas, (), { js_resizeCanvas(); } );

EM_JS( std::string, fs_read_file, (), { return FS.readFile( "Log.txt" ); } );

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_imgui;
EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_vfr;

void emscripten_loop()
try
{
    glfwPollEvents();
    global_window_handle->draw();
}
catch( const std::exception & e )
{
    fmt::print( "Caught std::exception in emscripten_loop! Message: {}\n", e.what() );
}
catch( ... )
{
    fmt::print( "Caught unknown exception in emscripten_loop!\n" );
}
#endif

static void framebuffer_size_callback( GLFWwindow * window, int width, int height )
{
    float xscale = 1;
    float yscale = 1;
#ifndef __EMSCRIPTEN__
    glfwGetWindowContentScale( window, &xscale, &yscale );
#endif
    global_window_handle->resize( width * xscale, height * yscale );
}

/////////////////////////////////////////////////////////////////////

namespace ui
{

static glm::vec2 interaction_click_pos;
static glm::vec2 mouse_pos_in_system;
static float radius_in_system;
static ImGuiID id_dockspace_left  = 0;
static ImGuiID id_dockspace_right = 0;

// Apply a callable to each of a variadic number of arguments
template<class F, class... Args>
void for_each_argument( F f, Args &&... args )
{
    []( ... ) {}( ( f( std::forward<Args>( args ) ), 0 )... );
}

/*
  Calculate coordinates relative to the system center from window device pixel coordinates.
    This assumes that `screen_pos` is relative to the top left corner of the window.
    `window_size` should be the device pixel size of the widget.
    This function also assumes the camera to be in an orthogonal z-projection.
*/
template<class... Args>
void transform_to_system_frame( RenderingLayer & rendering_layer, glm::vec2 window_size, Args &&... position )
{
    auto matrices
        = VFRendering::Utilities::getMatrices( rendering_layer.view.options(), window_size.x / window_size.y );
    auto camera_position = rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>();
    auto model_view      = glm::inverse( matrices.first );
    auto projection      = glm::inverse( matrices.second );

    for_each_argument(
        [&window_size, &camera_position, &projection, &model_view]( glm::vec2 & pos )
        {
            // Position relative to the center of the window [-1, 1]
            glm::vec2 relative_pos = 2.0f * ( pos - 0.5f * window_size );
            relative_pos.x /= window_size.x;
            relative_pos.y /= window_size.y;

            // Position projected into the system coordinate space
            glm::vec4 proj_back{ relative_pos.x, relative_pos.y, 0, 0 };
            proj_back = proj_back * projection;
            proj_back = proj_back * model_view;

            pos = glm::vec2{ proj_back.x + camera_position.x, -proj_back.y + camera_position.y };
        },
        position... );
}

void MainWindow::handle_mouse()
{
    auto & io = ImGui::GetIO();

#ifdef __APPLE__
    bool ctrl = io.KeySuper;
#else
    bool ctrl    = io.KeyCtrl;
#endif

#if defined( __APPLE__ )
    float scroll = -0.1 * io.MouseWheel;
#elif defined( __EMSCRIPTEN__ )
    float scroll = ( io.MouseWheel > 0 ? 1 : ( io.MouseWheel < 0 ? -1 : 0 ) );
#else
    float scroll = -io.MouseWheel;
#endif

    auto set_configuration = [&]()
    {
        scalar shift[3]{ interaction_click_pos.x - mouse_pos_in_system.x,
                         interaction_click_pos.y - mouse_pos_in_system.y, 0 };
        scalar current_position[3]{ mouse_pos_in_system.x, mouse_pos_in_system.y, 0 };
        scalar rect[3]{ -1, -1, -1 };

        Configuration_From_Clipboard_Shift( state.get(), shift, current_position, rect, radius_in_system );
    };

    auto set_atom_type = [&]( int atom_type )
    {
        scalar center[3];
        Geometry_Get_Center( this->state.get(), center );
        scalar current_position[3]{ mouse_pos_in_system.x, mouse_pos_in_system.y, 0.0f };
        current_position[0] -= center[0];
        current_position[1] -= center[1];
        scalar rect[3]{ -1, -1, -1 };

        Configuration_Set_Atom_Type( state.get(), atom_type, current_position, rect, radius_in_system );
    };

    auto set_pinning = [&]( bool pinned )
    {
        scalar center[3];
        Geometry_Get_Center( this->state.get(), center );
        scalar current_position[3]{ mouse_pos_in_system.x, mouse_pos_in_system.y, 0.0f };
        current_position[0] -= center[0];
        current_position[1] -= center[1];
        scalar rect[3]{ -1, -1, -1 };

        Configuration_Set_Pinned( state.get(), pinned, current_position, rect, radius_in_system );
    };

    if( !io.KeyShift )
        scroll *= 10;

    // Regular zoom
    if( io.MouseWheel && !ctrl )
    {
        rendering_layer.view.mouseScroll( scroll );
        rendering_layer.needs_redraw();
    }
    // Change interaction radius
    else if( io.MouseWheel && ctrl )
    {
        ui_config_file.interaction_radius += scroll;
        if( ui_config_file.interaction_radius < 0 )
            ui_config_file.interaction_radius = 0;
    }

    float scale = 1;
    if( io.KeyShift )
        scale = 0.1f;

    if( this->ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR )
    {
        mouse_pos_in_system = glm::vec2{ io.MousePos.x, io.MousePos.y };
        glm::vec2 radial_pos{ io.MousePos.x + ui_config_file.interaction_radius, io.MousePos.y };
        auto & io = ImGui::GetIO();
        glm::vec2 window_size
            = { ( 1 - left_sidebar_fraction - right_sidebar_fraction ) * io.DisplaySize.x, io.DisplaySize.y };
        transform_to_system_frame( rendering_layer, window_size, mouse_pos_in_system, radial_pos );
        radius_in_system = radial_pos.x - mouse_pos_in_system.x;
    }

    // Left-click in interaction mode
    if( ImGui::IsMouseClicked( GLFW_MOUSE_BUTTON_LEFT, false )
        && this->ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR )
    {
        interaction_click_pos = mouse_pos_in_system;

        if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
            Configuration_To_Clipboard( state.get() );
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
            set_atom_type( -1 );
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
            set_pinning( true );
    }
    // Right-click in interaction mode
    if( ImGui::IsMouseClicked( GLFW_MOUSE_BUTTON_RIGHT, false ) )
    {
        if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
            set_configuration();
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
            set_atom_type( 0 );
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
            set_pinning( false );
    }

    // Left-click dragging
    if( ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT )
        && this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR )
    {
        rendering_layer.view.mouseMove(
            glm::vec2( 0, 0 ), glm::vec2( scale * io.MouseDelta.x, scale * io.MouseDelta.y ),
            VFRendering::CameraMovementModes::ROTATE_BOUNDED );
        rendering_layer.needs_redraw();
    }
    else if(
        ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT )
        && this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
    {
        set_configuration();
    }
    else if(
        ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT )
        && this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
    {
        set_atom_type( -1 );
    }
    else if(
        ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT )
        && this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
    {
        set_pinning( true );
    }
    // Right-click dragging
    else if( ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) )
    {
        if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR
            || this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
        {
            rendering_layer.view.mouseMove(
                glm::vec2( 0, 0 ), glm::vec2( scale * io.MouseDelta.x, scale * io.MouseDelta.y ),
                VFRendering::CameraMovementModes::TRANSLATE );
            rendering_layer.needs_redraw();
        }
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
        {
            set_atom_type( 0 );
        }
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
        {
            set_pinning( false );
        }
    }
}

void MainWindow::handle_keyboard()
{
    auto & io = ImGui::GetIO();

#ifdef __APPLE__
    bool ctrl = io.KeySuper;
#else
    bool ctrl    = io.KeyCtrl;
#endif

    if( ctrl && io.KeyShift )
    {
        if( ImGui::IsKeyPressed( ImGuiKey_R ) )
        {
            this->rendering_layer.reset_camera();
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F, false ) )
        {
            fullscreen_toggled = true;
        }
    }
    else if( ctrl )
    {
        if( ImGui::IsKeyPressed( ImGuiKey_R ) )
        {
            Configuration_Random( state.get() );
            rendering_layer.needs_data();
            this->ui_shared_state.notify( "Randomized spins" );
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F, false ) )
        {
            ui_config_file.window_hide_menubar = !ui_config_file.window_hide_menubar;
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( ImGuiKey_X ) )
        {
            this->cut_image();
        }
        if( ImGui::IsKeyPressed( ImGuiKey_C ) )
        {
            Chain_Image_to_Clipboard( global_window_handle->state.get() );
        }
        if( ImGui::IsKeyPressed( ImGuiKey_V ) )
        {
            this->paste_image();
        }
        if( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
        {
            this->insert_image_left();
        }
        if( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
        {
            this->insert_image_right();
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( ImGuiKey_N ) )
        {
            auto & conf = ui_shared_state.configurations;

            Configuration_Add_Noise_Temperature(
                state.get(), conf.noise_temperature, conf.pos, conf.border_rect, conf.border_cyl, conf.border_sph,
                conf.inverted );

            Chain_Update_Data( state.get() );
            rendering_layer.needs_data();
        }
    }
    else
    {
        //-----------------------------------------------------
        // Keyboard camera controls

        // Faster repeat rate for smooth camera
        const float backup_repeat_delay = io.KeyRepeatDelay; // default 0.250f
        const float backup_repeat_rate  = io.KeyRepeatRate;  // default 0.050f
        io.KeyRepeatDelay               = 0.01f;
        io.KeyRepeatRate                = 0.01f;

        float scale = 1;
        if( io.KeyShift )
            scale = 0.1f;

        if( ImGui::IsKeyDown( ImGuiKey_W ) && !ImGui::IsKeyDown( ImGuiKey_S ) )
        {
            rendering_layer.view.mouseScroll( -scale );
            rendering_layer.needs_redraw();
        }
        else if( ImGui::IsKeyDown( ImGuiKey_S ) && !ImGui::IsKeyDown( ImGuiKey_W ) )
        {
            rendering_layer.view.mouseScroll( scale );
            rendering_layer.needs_redraw();
        }

        bool rotate_camera = false;
        bool move_camera   = false;
        float dx           = 0;
        float dy           = 0;
        float theta        = 0;
        float phi          = 0;

        if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR )
        {
            if( ImGui::IsKeyDown( ImGuiKey_A ) && !ImGui::IsKeyDown( ImGuiKey_D ) )
            {
                rotate_camera = true;
                phi           = 5 * scale;
            }
            else if( ImGui::IsKeyDown( ImGuiKey_D ) && !ImGui::IsKeyDown( ImGuiKey_A ) )
            {
                rotate_camera = true;
                phi           = -5 * scale;
            }

            if( ImGui::IsKeyDown( ImGuiKey_Q ) && !ImGui::IsKeyDown( ImGuiKey_E ) )
            {
                rotate_camera = true;
                theta         = 5 * scale;
            }
            else if( ImGui::IsKeyDown( ImGuiKey_E ) && !ImGui::IsKeyDown( ImGuiKey_Q ) )
            {
                rotate_camera = true;
                theta         = -5 * scale;
            }
        }

        if( ImGui::IsKeyDown( ImGuiKey_T ) && !ImGui::IsKeyDown( ImGuiKey_G ) )
        {
            move_camera = true;
            dy          = scale;
        }
        else if( ImGui::IsKeyDown( ImGuiKey_G ) && !ImGui::IsKeyDown( ImGuiKey_T ) )
        {
            move_camera = true;
            dy          = -scale;
        }

        if( ImGui::IsKeyDown( ImGuiKey_F ) && !ImGui::IsKeyDown( ImGuiKey_H ) )
        {
            move_camera    = true;
            dx             = scale;
        }
        else if( ImGui::IsKeyDown( ImGuiKey_H ) && !ImGui::IsKeyDown( ImGuiKey_F ) )
        {
            move_camera = true;
            dx          = -scale;
        }

        if( rotate_camera )
        {
            rendering_layer.view.mouseMove(
                { 0, 0 }, { phi, theta }, VFRendering::CameraMovementModes::ROTATE_BOUNDED );
            rendering_layer.needs_redraw();
        }
        if( move_camera )
        {
            rendering_layer.view.mouseMove( { 0, 0 }, { dx, dy }, VFRendering::CameraMovementModes::TRANSLATE );
            rendering_layer.needs_redraw();
        }

        // Reset the key repeat parameters
        io.KeyRepeatRate  = backup_repeat_rate;
        io.KeyRepeatDelay = backup_repeat_delay;

        if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR )
        {
            if( ImGui::IsKeyPressed( ImGuiKey_X, false ) )
            {
                float camera_distance = glm::length(
                    rendering_layer.view.options().get<VFRendering::View::Option::CENTER_POSITION>()
                    - rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>() );
                auto center_position = rendering_layer.view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
                auto camera_position = center_position;
                auto up_vector       = glm::vec3( 0, 0, 1 );

                if( !io.KeyShift )
                {
                    camera_position += camera_distance * glm::vec3( 1, 0, 0 );
                }
                else
                {
                    camera_position -= camera_distance * glm::vec3( 1, 0, 0 );
                }

                VFRendering::Options options;
                options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
                options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
                options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
                rendering_layer.view.updateOptions( options );
                rendering_layer.needs_redraw();
            }
            if( ImGui::IsKeyPressed( ImGuiKey_Y, false ) )
            {
                float camera_distance = glm::length(
                    rendering_layer.view.options().get<VFRendering::View::Option::CENTER_POSITION>()
                    - rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>() );
                auto center_position = rendering_layer.view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
                auto camera_position = center_position;
                auto up_vector       = glm::vec3( 0, 0, 1 );

                if( !io.KeyShift )
                    camera_position += camera_distance * glm::vec3( 0, -1, 0 );
                else
                    camera_position -= camera_distance * glm::vec3( 0, -1, 0 );

                VFRendering::Options options;
                options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
                options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
                options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
                rendering_layer.view.updateOptions( options );
                rendering_layer.needs_redraw();
            }
            if( ImGui::IsKeyPressed( ImGuiKey_Z, false ) )
            {
                float camera_distance = glm::length(
                    rendering_layer.view.options().get<VFRendering::View::Option::CENTER_POSITION>()
                    - rendering_layer.view.options().get<VFRendering::View::Option::CAMERA_POSITION>() );
                auto center_position = rendering_layer.view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
                auto camera_position = center_position;
                auto up_vector       = glm::vec3( 0, 1, 0 );

                if( !io.KeyShift )
                    camera_position += camera_distance * glm::vec3( 0, 0, 1 );
                else
                    camera_position -= camera_distance * glm::vec3( 0, 0, 1 );

                VFRendering::Options options;
                options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
                options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
                options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
                rendering_layer.view.updateOptions( options );
                rendering_layer.needs_redraw();
            }

            if( ImGui::IsKeyPressed( ImGuiKey_C, false ) )
            {
                rendering_layer.set_camera_orthographic( !ui_shared_state.camera_is_orthographic );
            }
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( ImGuiKey_Space, false ) )
        {
            start_stop();
        }

        if( ImGui::IsKeyPressed( ImGuiKey_RightArrow ) )
        {
            if( System_Get_Index( state.get() ) < Chain_Get_NOI( this->state.get() ) - 1 )
            {
                Chain_next_Image( this->state.get() );
                rendering_layer.needs_data();
                parameters_widget.update_data();
            }
        }

        if( ImGui::IsKeyPressed( ImGuiKey_LeftArrow ) )
        {
            if( System_Get_Index( state.get() ) > 0 )
            {
                Chain_prev_Image( this->state.get() );
                rendering_layer.needs_data();
                parameters_widget.update_data();
            }
        }

        if( ImGui::IsKeyPressed( ImGuiKey_Delete ) )
        {
            this->delete_image();
        }

        if( ImGui::IsKeyPressed( ImGuiKey_I, false ) )
        {
            ui_config_file.show_overlays = !ui_config_file.show_overlays;
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( ImGuiKey_F1, false ) )
        {
            show_keybindings = !show_keybindings;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F2, false ) )
        {
            ui_config_file.show_settings = !ui_config_file.show_settings;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F3, false ) )
        {
            ui_config_file.show_plots = !ui_config_file.show_plots;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F4, false ) )
        {
            ui_config_file.show_visualisation_widget = !ui_config_file.show_visualisation_widget;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F5, false ) )
        {
            if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
                this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );
            else
                this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::DRAG );
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F6, false ) )
        {
            if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
                this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );
            else
                this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::DEFECT );
        }
        if( ImGui::IsKeyPressed( ImGuiKey_F7, false ) )
        {
            if( SPIRIT_PINNING_ENABLED )
            {
                if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );
                else
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::PINNING );
            }
            else
            {
                ui_shared_state.notify( ICON_FA_EXCLAMATION_TRIANGLE " pinning is not enabled" );
            }
        }

        //-----------------------------------------------------
        // TODO: deactivate method selection if a calculation is running
        if( ImGui::IsKeyPressed( ImGuiKey_1, false ) )
        {
            ui_shared_state.selected_mode = GUI_Mode::Minimizer;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_2, false ) )
        {
            ui_shared_state.selected_mode = GUI_Mode::MC;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_3, false ) )
        {
            ui_shared_state.selected_mode = GUI_Mode::LLG;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_4, false ) )
        {
            ui_shared_state.selected_mode = GUI_Mode::GNEB;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_5, false ) )
        {
            ui_shared_state.selected_mode = GUI_Mode::MMF;
        }
        if( ImGui::IsKeyPressed( ImGuiKey_6, false ) )
        {
            ui_shared_state.selected_mode = GUI_Mode::EMA;
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( ImGuiKey_F10, false ) )
        {
            ui_config_file.window_hide_menubar = !ui_config_file.window_hide_menubar;
        }

        if( ImGui::IsKeyPressed( ImGuiKey_F11, false ) )
        {
            ui_config_file.window_fullscreen = !ui_config_file.window_fullscreen;
        }

        if( ImGui::IsKeyPressed( ImGuiKey_Home, false ) || ImGui::IsKeyPressed( ImGuiKey_F12, false ) )
        {
            ++ui_shared_state.n_screenshots;
            std::string name
                = fmt::format( "{}_Screenshot_{}", State_DateTime( state.get() ), ui_shared_state.n_screenshots );
            rendering_layer.screenshot_png( name );
            ui_shared_state.notify( fmt::format( ICON_FA_DESKTOP "  Captured \"{}\"", name ), 4 );
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( ImGuiKey_Enter, false ) )
        {
            if( ui_shared_state.configurations.last_used != "" )
            {
                Log_Send(
                    state.get(), Log_Level_Debug, Log_Sender_UI,
                    ( "Inserting last used configuration: " + ui_shared_state.configurations.last_used ).c_str() );

                if( ui_shared_state.configurations.last_used == "plus_z" )
                    this->configurations_widget.set_plus_z();
                else if( ui_shared_state.configurations.last_used == "minus_z" )
                    this->configurations_widget.set_minus_z();
                else if( ui_shared_state.configurations.last_used == "random" )
                    this->configurations_widget.set_random();
                else if( ui_shared_state.configurations.last_used == "spiral" )
                    this->configurations_widget.set_spiral();
                else if( ui_shared_state.configurations.last_used == "skyrmion" )
                    this->configurations_widget.set_skyrmion();
                else if( ui_shared_state.configurations.last_used == "hopfion" )
                    this->configurations_widget.set_hopfion();
            }
        }
    }
}

void MainWindow::start_stop()
try
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Start/Stop" );

    Chain_Update_Data( this->state.get() );

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() ) )
    {
        // Running, so we stop it
        Simulation_Stop( this->state.get() );
        // Join the thread of the stopped simulation
        if( threads_image[System_Get_Index( state.get() )].joinable() )
            threads_image[System_Get_Index( state.get() )].join();
        else if( thread_chain.joinable() )
            thread_chain.join();
        this->ui_shared_state.notify( "stopped calculation" );
        calculation_running = false;
    }
    else
    {
        // Not running, so we start it
        if( ui_shared_state.selected_mode == GUI_Mode::Minimizer )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )] = std::thread(
                &Simulation_LLG_Start, this->state.get(), ui_shared_state.selected_solver_min, -1, -1, false, nullptr,
                -1, -1 );
        }
        if( ui_shared_state.selected_mode == GUI_Mode::LLG )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )] = std::thread(
                &Simulation_LLG_Start, this->state.get(), ui_shared_state.selected_solver_llg, -1, -1, false, nullptr,
                -1, -1 );
        }
        else if( ui_shared_state.selected_mode == GUI_Mode::MC )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_MC_Start, this->state.get(), -1, -1, false, nullptr, -1, -1 );
        }
        else if( ui_shared_state.selected_mode == GUI_Mode::GNEB )
        {
            if( thread_chain.joinable() )
                thread_chain.join();
            this->thread_chain = std::thread(
                &Simulation_GNEB_Start, this->state.get(), ui_shared_state.selected_solver_min, -1, -1, false, nullptr,
                -1 );
        }
        else if( ui_shared_state.selected_mode == GUI_Mode::MMF )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )] = std::thread(
                &Simulation_MMF_Start, this->state.get(), ui_shared_state.selected_solver_min, -1, -1, false, nullptr,
                -1, -1 );
        }
        else if( ui_shared_state.selected_mode == GUI_Mode::EMA )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_EMA_Start, this->state.get(), -1, -1, false, nullptr, -1, -1 );
        }
        this->ui_shared_state.notify( "started calculation" );
    }
    rendering_layer.needs_data();
}
catch( const std::exception & e )
{
    Log_Send(
        state.get(), Log_Level_Error, Log_Sender_UI, fmt::format( "caught std::exception: {}\n", e.what() ).c_str() );
}

void MainWindow::stop_all()
try
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Stopping all calculations" );

    Simulation_Stop_All( state.get() );

    for( unsigned int i = 0; i < threads_image.size(); ++i )
    {
        if( threads_image[i].joinable() )
            threads_image[i].join();
    }
    if( thread_chain.joinable() )
        thread_chain.join();

    this->ui_shared_state.notify( "stopped all calculations" );

    rendering_layer.needs_data();
}
catch( const std::exception & e )
{
    Log_Send(
        state.get(), Log_Level_Error, Log_Sender_UI, fmt::format( "caught std::exception: {}\n", e.what() ).c_str() );
}

void MainWindow::stop_current()
try
{
    Log_Send( state.get(), Log_Level_Debug, Log_Sender_UI, "Stopping current calculation" );

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() ) )
    {
        // Running, so we stop it
        Simulation_Stop( this->state.get() );
        // Join the thread of the stopped simulation
        if( threads_image[System_Get_Index( state.get() )].joinable() )
            threads_image[System_Get_Index( state.get() )].join();
        else if( thread_chain.joinable() )
            thread_chain.join();
    }

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() ) )
    {
        // Running, so we stop it
        Simulation_Stop( this->state.get() );
        // Join the thread of the stopped simulation
        if( threads_image[System_Get_Index( state.get() )].joinable() )
            threads_image[System_Get_Index( state.get() )].join();
        else if( thread_chain.joinable() )
            thread_chain.join();
    }

    this->ui_shared_state.notify( "stopped current calculation" );
    rendering_layer.needs_data();
}
catch( const std::exception & e )
{
    Log_Send(
        state.get(), Log_Level_Error, Log_Sender_UI, fmt::format( "caught std::exception: {}\n", e.what() ).c_str() );
}

void MainWindow::cut_image()
{
    if( Chain_Get_NOI( state.get() ) > 1 )
    {
        stop_current();

        Chain_Image_to_Clipboard( state.get() );

        int idx = System_Get_Index( state.get() );
        if( Chain_Delete_Image( state.get(), idx ) )
        {
            // Make the threads_image vector smaller
            if( this->threads_image[idx].joinable() )
                this->threads_image[idx].join();
            this->threads_image.erase( threads_image.begin() + idx );

            this->ui_shared_state.notify( fmt::format( "cut image {}", idx + 1 ) );
        }
        rendering_layer.needs_data();
    }
}

void MainWindow::paste_image()
{
    // Paste a Spin System into current System
    this->stop_current();
    Chain_Replace_Image( state.get() );
    rendering_layer.needs_data();

    this->ui_shared_state.notify( "pasted image from clipboard" );
}

void MainWindow::insert_image_left()
{
    if( Simulation_Running_On_Chain( state.get() ) )
        this->stop_current();

    int idx = System_Get_Index( state.get() );
    // Insert Image
    Chain_Insert_Image_Before( state.get() );
    // Make the threads_image vector larger
    this->threads_image.insert( threads_image.begin() + idx, std::thread() );
    // Switch to the inserted image
    Chain_prev_Image( this->state.get() );
    rendering_layer.needs_data();
    parameters_widget.update_data();

    this->ui_shared_state.notify( "inserted image to the left" );
}

void MainWindow::insert_image_right()
{
    if( Simulation_Running_On_Chain( state.get() ) )
        this->stop_current();

    int idx = System_Get_Index( state.get() );
    // Insert Image
    Chain_Insert_Image_After( state.get() );
    // Make the threads_image vector larger
    this->threads_image.insert( threads_image.begin() + idx + 1, std::thread() );
    // Switch to the inserted image
    Chain_next_Image( this->state.get() );
    rendering_layer.needs_data();
    parameters_widget.update_data();

    this->ui_shared_state.notify( "inserted image to the right" );
}

void MainWindow::delete_image()
{
    if( Chain_Get_NOI( state.get() ) > 1 )
    {
        this->stop_current();

        int idx = System_Get_Index( state.get() );
        if( Chain_Delete_Image( state.get() ) )
        {
            // Make the threads_image vector smaller
            if( this->threads_image[idx].joinable() )
                this->threads_image[idx].join();
            this->threads_image.erase( threads_image.begin() + idx );

            this->ui_shared_state.notify( fmt::format( "deleted image {}", idx + 1 ) );
        }

        rendering_layer.needs_data();
    }
}

void MainWindow::draw()
{
#ifndef __EMSCRIPTEN__
    if( fullscreen_toggled )
    {
        if( ui_config_file.window_fullscreen )
        {
            auto pos  = ui_config_file.window_position.data();
            auto size = ui_config_file.window_size.data();
            glfwSetWindowMonitor( glfw_window, nullptr, pos[0], pos[1], size[0], size[1], 0 );
            ui_config_file.window_fullscreen = false;
            this->ui_shared_state.notify( "Exited fullscreen" );
        }
        else
        {
            GLFWmonitor * monitor    = glfwGetPrimaryMonitor();
            const GLFWvidmode * mode = glfwGetVideoMode( monitor );
            glfwSetWindowMonitor( glfw_window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate );
            ui_config_file.window_fullscreen = true;
            this->ui_shared_state.notify( "Entered fullscreen. ctrl+shift+f to exit", 5 );
        }
        fullscreen_toggled = false;
    }
#endif

    int display_w, display_h;
#ifdef __EMSCRIPTEN__
    display_w = canvas_get_width();
    display_h = canvas_get_height();
    glfwSetWindowSize( glfw_window, display_w, display_h );
#else
    glfwMakeContextCurrent( glfw_window );
    glfwGetFramebufferSize( glfw_window, &display_w, &display_h );
#endif

#ifdef __EMSCRIPTEN__
    emscripten_webgl_make_context_current( context_imgui );
    glViewport( 0, 0, display_w, display_h );
    glClearColor( 0, 0, 0, 0 );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
#endif

    auto & io = ImGui::GetIO();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // ----------------

    if( !io.WantCaptureMouse )
        this->handle_mouse();

    if( !io.WantCaptureKeyboard )
        this->handle_keyboard();

    draw_imgui( display_w, display_h );

    glfwSwapBuffers( glfw_window );

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() )
        || ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR )
    {
        rendering_layer.needs_data();
    }

#ifdef __EMSCRIPTEN__
    emscripten_webgl_make_context_current( context_vfr );
#endif
    rendering_layer.draw( display_w, display_h );
}

void MainWindow::show_dock_widgets()
{
    auto & io = ImGui::GetIO();

    // Save references to the widgets in a vector so we can iterate over them
    static std::array<WidgetBase *, 6> spirit_widgets
        = { &configurations_widget, &parameters_widget, &hamiltonian_widget,
            &geometry_widget,       &plots_widget,      &visualisation_widget };

    static bool previous_frame_showed_left_dock  = false;
    static bool previous_frame_showed_right_dock = false;
    static bool show_left_dock_agein             = true;
    static bool show_right_dock_agein            = true;

    bool dragging_a_window        = false;
    bool a_window_wants_to_dock   = false;
    bool a_window_is_docked_left  = false;
    bool a_window_is_docked_right = false;

    // Temporaries to compare against later
    float pre_right_sidebar_fraction = right_sidebar_fraction;
    float pre_left_sidebar_fraction  = left_sidebar_fraction;

    auto window_height  = io.DisplaySize.y;
    auto sidebar_height = window_height - menu_bar_size[1];

    for( const auto * w : spirit_widgets )
    {
        if( !w->show_ )
            continue;

        if( w->dragging )
            dragging_a_window = true;

        if( w->wants_to_dock )
            a_window_wants_to_dock = true;

        if( previous_frame_showed_left_dock )
        {
            if( w->root_dock_node_id == id_dockspace_left )
                a_window_is_docked_left = true;
        }

        if( previous_frame_showed_right_dock )
        {
            if( w->root_dock_node_id == id_dockspace_right )
                a_window_is_docked_right = true;
        }
    }

    auto show_left_dock = [&]()
    {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
                                        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking
                                        | ImGuiWindowFlags_NoScrollbar;
        ImGuiDockNodeFlags dock_flags = ImGuiDockNodeFlags_None;

        ImGui::SetNextWindowPos( { 0, menu_bar_size[1] } );
        if( a_window_is_docked_left )
        {
            ImGui::SetNextWindowSizeConstraints(
                { sidebar_fraction_min * io.DisplaySize.x, sidebar_height },
                { sidebar_fraction_max * io.DisplaySize.x, sidebar_height } );
        }
        else
        {
            ImGui::SetNextWindowSize(
                { sidebar_fraction_default * io.DisplaySize.x, io.DisplaySize.y - menu_bar_size[1] } );
            ImGui::SetNextWindowBgAlpha( 0.5f );
            window_flags |= ImGuiWindowFlags_NoResize;
        }

        ImGui::Begin( "##sidebar_left", nullptr, window_flags );

        if( a_window_is_docked_left )
        {
            left_sidebar_fraction = ImGui::GetWindowSize()[0] / io.DisplaySize.x;
        }
        else
        {
            left_sidebar_fraction = 0;
        }

        id_dockspace_left = ImGui::GetID( "dockspace_left" );
        ImGui::DockSpace( id_dockspace_left, { -1, -1 }, dock_flags );

        ImGui::End();
    };

    auto show_right_dock = [&]()
    {
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse
                                        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDocking
                                        | ImGuiWindowFlags_NoScrollbar;
        ImGuiDockNodeFlags dock_flags = ImGuiDockNodeFlags_None;

        if( a_window_is_docked_right )
        {
            ImGui::SetNextWindowPos( { ( 1 - right_sidebar_fraction ) * io.DisplaySize.x, menu_bar_size[1] } );
            ImGui::SetNextWindowSizeConstraints(
                { sidebar_fraction_min * io.DisplaySize.x, sidebar_height },
                { sidebar_fraction_max * io.DisplaySize.x, sidebar_height } );
        }
        else
        {
            ImGui::SetNextWindowPos( { ( 1 - sidebar_fraction_default ) * io.DisplaySize.x, menu_bar_size[1] } );
            ImGui::SetNextWindowSize(
                { sidebar_fraction_default * io.DisplaySize.x, io.DisplaySize.y - menu_bar_size[1] } );
            ImGui::SetNextWindowBgAlpha( 0.5f );
            window_flags |= ImGuiWindowFlags_NoResize;
        }

        ImGui::Begin( "##sidebar_right", nullptr, window_flags );

        if( a_window_is_docked_right )
        {
            right_sidebar_fraction = ImGui::GetWindowSize()[0] / io.DisplaySize.x;
        }
        else
        {
            right_sidebar_fraction = 0;
        }

        id_dockspace_right = ImGui::GetID( "dockspace_right" );
        ImGui::DockSpace( id_dockspace_right, { -1, -1 }, dock_flags );

        ImGui::End();
    };

    if( a_window_is_docked_left || dragging_a_window || a_window_wants_to_dock )
    {
        show_left_dock();
        previous_frame_showed_left_dock = true;
        show_left_dock_agein            = true;
    }
    else if( show_left_dock_agein )
    {
        show_left_dock();
        previous_frame_showed_left_dock = true;
        show_left_dock_agein            = false;
    }
    else
        previous_frame_showed_left_dock = false;

    if( a_window_is_docked_right || dragging_a_window || a_window_wants_to_dock )
    {
        show_right_dock();
        previous_frame_showed_right_dock = true;
        show_right_dock_agein            = true;
    }
    else if( show_right_dock_agein )
    {
        show_right_dock();
        previous_frame_showed_right_dock = true;
        show_right_dock_agein            = false;
    }
    else
        previous_frame_showed_right_dock = false;

    // If the sidebar changed we need to issue a redraw and resize
    if( ( left_sidebar_fraction != pre_left_sidebar_fraction )
        || ( right_sidebar_fraction != pre_right_sidebar_fraction ) )
    {
        rendering_layer.needs_redraw();

        float layout_left  = 0;
        float layout_right = 1;

        if( a_window_is_docked_left )
            layout_left = left_sidebar_fraction;
        if( a_window_is_docked_right )
            layout_right = 1 - right_sidebar_fraction;

        rendering_layer.rendering_layout = { layout_left, 0, layout_right - layout_left, 1 };

        // Set new rendering position/size
        rendering_layer.update_renderers_from_layout();
    }
}

void MainWindow::draw_imgui( int display_w, int display_h )
{
    auto & io    = ImGui::GetIO();
    auto & style = ImGui::GetStyle();

    if( ui_shared_state.interaction_mode != UiSharedState::InteractionMode::REGULAR )
    {
        if( ImGui::IsWindowHovered( ImGuiHoveredFlags_AnyWindow ) )
            glfwSetInputMode( glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL );
        else
        {
            glfwSetInputMode( glfw_window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN );

            ImU32 color = IM_COL32( 0, 0, 0, 255 );
            if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
                color = IM_COL32( 255, 0, 0, 100 );
            else if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
                color = IM_COL32( 0, 255, 0, 100 );
            else if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
                color = IM_COL32( 0, 0, 255, 100 );

            ImGui::GetBackgroundDrawList()->AddCircle(
                ImGui::GetMousePos(), ui_config_file.interaction_radius, color, 0, 4 );
            ImGui::GetBackgroundDrawList()->AddCircleFilled( ImGui::GetMousePos(), 8, color );
        }

        std::string title_text;
        if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
            title_text = "Interaction mode: dragging";
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
            title_text = "Interaction mode: defects";
        else if( this->ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
            title_text = "Interaction mode: pinning";
        ImVec2 title_text_size = ImGui::CalcTextSize( title_text.c_str() );

        std::string mouse_text = fmt::format(
            "Mouse is at system position ({:.3f}, {:.3f})", mouse_pos_in_system.x - geometry_widget.system_center[0],
            mouse_pos_in_system.y - geometry_widget.system_center[1] );
        ImVec2 mouse_text_size = ImGui::CalcTextSize( mouse_text.c_str() );

        std::string radius_text = fmt::format( "radius = {:.3f}", radius_in_system );
        ImVec2 radius_text_size = ImGui::CalcTextSize( radius_text.c_str() );

        const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                              | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                              | ImGuiWindowFlags_NoNav;

        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0 );
        ImGui::PushStyleVar( ImGuiStyleVar_Alpha, 0.6f );

        ImGui::SetNextWindowPos( { 0.5f * ( io.DisplaySize.x - mouse_text_size.x ), 45 } );
        // Also need to set size, because window may otherwise flicker for some reason...
        ImGui::SetNextWindowSize( { mouse_text_size.x + 2 * style.WindowPadding.x,
                                    title_text_size.y + mouse_text_size.y + radius_text_size.y
                                        + 2 * style.FramePadding.y + 2 * style.WindowPadding.y } );

        if( ImGui::Begin( "##mouse_system_pos", nullptr, window_flags ) )
        {
            ImGui::Dummy( { 0.5f * ( mouse_text_size.x - title_text_size.x ), 0 } );
            ImGui::SameLine();
            ImGui::TextUnformatted( title_text.c_str() );
            ImGui::TextUnformatted( mouse_text.c_str() );
            ImGui::Dummy( { 0.5f * ( mouse_text_size.x - radius_text_size.x ), 0 } );
            ImGui::SameLine();
            ImGui::TextUnformatted( radius_text.c_str() );
            ImGui::End();
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleVar();
    }
    else
        glfwSetInputMode( glfw_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL );

    // ----------------

    ImGui::PushFont( font_karla_14 );

    this->show_menu_bar();
    this->show_notifications();
    this->show_dock_widgets();

    // ----------------

    ImVec2 viewport_pos  = { left_sidebar_fraction * io.DisplaySize.x, menu_bar_size[1] };
    ImVec2 viewport_size = { ( 1 - left_sidebar_fraction - right_sidebar_fraction ) * io.DisplaySize.x, -1 };
    auto overlay_pos     = ui_config_file.overlay_system_position;

    ImGui::PushFont( font_cousine_14 );
    widgets::show_overlay_system(
        ui_config_file.show_overlays, ui_config_file.overlay_system_corner, overlay_pos, state, viewport_pos,
        viewport_size );
    widgets::show_overlay_calculation(
        ui_config_file.show_overlays, ui_shared_state.selected_mode, ui_shared_state.selected_solver_min,
        ui_shared_state.selected_solver_llg, ui_config_file.overlay_calculation_corner,
        ui_config_file.overlay_calculation_position, state, viewport_pos, viewport_size );
    ImGui::PopFont();

    // ----------------

    this->configurations_widget.show();
    this->parameters_widget.show();
    this->hamiltonian_widget.show();
    ImGui::PushFont( font_cousine_14 );
    this->geometry_widget.show();
    this->plots_widget.show();
    ImGui::PopFont();
    this->visualisation_widget.show();

    widgets::show_settings( ui_config_file.show_settings, rendering_layer );
    widgets::show_keybindings( show_keybindings );
    widgets::show_about( show_about );

    if( show_imgui_demo_window )
        ImGui::ShowDemoWindow( &show_imgui_demo_window );
    if( show_implot_demo_window )
        ImPlot::ShowDemoWindow( &show_implot_demo_window );

    ImGui::PopFont();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}

void MainWindow::show_menu_bar()
{
    static auto modes = std::map<GUI_Mode, std::pair<std::string, std::string>>{
        { GUI_Mode::Minimizer, { "Minimizer", "(1) energy minimisation" } },
        { GUI_Mode::MC, { "Monte Carlo", "(2) Monte Carlo Stochastical sampling" } },
        { GUI_Mode::LLG, { "LLG", "(3) Landau-Lifshitz-Gilbert dynamics" } },
        { GUI_Mode::GNEB, { "GNEB", "(4) geodesic nudged elastic band calculation" } },
        { GUI_Mode::MMF, { "MMF", "(5) minimum mode following saddle point search" } },
        { GUI_Mode::EMA, { "Eigenmodes", "(6) eigenmode calculation and visualisation" } }
    };
    static std::vector<float> mode_button_hovered_duration( modes.size(), 0 );
    static auto image_number = static_cast<ImU32>( 1 );
    static auto chain_length = static_cast<ImU32>( 1 );

    static bool previous_frame_menu_was_active = false;

    // Always show the menu bar if the mouse is near the top or a menu item is active
    if( ui_config_file.window_hide_menubar
        && !( ImGui::GetMousePos().y < menu_bar_size[1] || previous_frame_menu_was_active ) )
        return;

    // ImGui::PushFont( font_karla_16 );
    float font_size_px             = font_karla_14->FontSize;
    previous_frame_menu_was_active = false;

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 5.f, 5.f ) );
    if( ImGui::BeginMainMenuBar() )
    {
        ImGui::PopStyleVar();

        if( ImGui::BeginMenu( "File" ) )
        {
            previous_frame_menu_was_active = true;
            if( ImGui::MenuItem( "Load config-file" ) )
            {
                nfdpathset_t path_set;
                nfdresult_t result = NFD_OpenDialogMultiple( "cfg", NULL, &path_set );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &path_set ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &path_set, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &path_set );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save current cfg" ) )
            {
                nfdchar_t * save_path = NULL;
                nfdresult_t result    = NFD_SaveDialog( "cfg", NULL, &save_path );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", save_path );
                    free( save_path );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Load spin configuration" ) )
            {
                nfdpathset_t path_set;
                nfdresult_t result = NFD_OpenDialogMultiple( "ovf;txt;csv", NULL, &path_set );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &path_set ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &path_set, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &path_set );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save spin configuration" ) )
            {
                nfdchar_t * save_path = NULL;
                nfdresult_t result    = NFD_SaveDialog( "ovf;txt;csv", NULL, &save_path );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", save_path );
                    free( save_path );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Load system eigenmodes", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Save system eigenmodes", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Save energy per spin", "", false, false ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Load chain" ) )
            {
                nfdpathset_t path_set;
                nfdresult_t result = NFD_OpenDialogMultiple( "ovf;txt;csv", NULL, &path_set );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &path_set ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &path_set, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &path_set );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save chain" ) )
            {
                nfdchar_t * save_path = NULL;
                nfdresult_t result    = NFD_SaveDialog( "ovf;txt;csv", NULL, &save_path );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", save_path );
                    free( save_path );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save energies", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Save interpolated energies", "", false, false ) )
            {
            }
            ImGui::Separator();
            ImGui::MenuItem( ICON_FA_COG "  Settings", "F2", &ui_config_file.show_settings );
            ImGui::Separator();
            if( ImGui::MenuItem( ICON_FA_DESKTOP "  Take screenshot", "F12, " ICON_FA_HOME ) )
            {
                ++ui_shared_state.n_screenshots;
                std::string name
                    = fmt::format( "{}_Screenshot_{}", State_DateTime( state.get() ), ui_shared_state.n_screenshots );
                rendering_layer.screenshot_png( name );
                ui_shared_state.notify( fmt::format( ICON_FA_DESKTOP "  Captured \"{}\"", name ), 4 );
            }
            if( ImGui::MenuItem( ICON_FA_DESKTOP "  Take screenshots of the chain" ) )
            {
                int display_w, display_h;
                glfwGetFramebufferSize( glfw_window, &display_w, &display_h );
                int active_image = System_Get_Index( this->state.get() );
                State_DateTime( state.get() );
                Chain_Jump_To_Image( this->state.get(), 0 );
                std::string tag = State_DateTime( state.get() );
                ++ui_shared_state.n_screenshots_chain;
                for( int i = 0; i < Chain_Get_NOI( this->state.get() ); ++i )
                {
                    std::string name
                        = fmt::format( "{}_Screenshot_Chain_{}_Image_{}", tag, ui_shared_state.n_screenshots, i );

                    rendering_layer.needs_data();
                    rendering_layer.draw( display_w, display_h );

                    rendering_layer.screenshot_png( name );
                    Chain_next_Image( this->state.get() );
                }
                ui_shared_state.notify(
                    fmt::format(
                        ICON_FA_DESKTOP "  Captured series \"{}_Screenshot_Chain_{}_Image_...\"", tag,
                        ui_shared_state.n_screenshots ),
                    4 );
                Chain_Jump_To_Image( this->state.get(), active_image );
                rendering_layer.needs_data();
                rendering_layer.draw( display_w, display_h );
            }

            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Edit" ) )
        {
            previous_frame_menu_was_active = true;
            if( ImGui::MenuItem( ICON_FA_CUT "  Cut system", "ctrl+x" ) )
            {
                this->cut_image();
            }
            if( ImGui::MenuItem( ICON_FA_COPY "  Copy system", "ctrl+c" ) )
            {
                Chain_Image_to_Clipboard( this->state.get() );
            }
            if( ImGui::MenuItem( ICON_FA_PASTE "  Paste system", "ctrl+v" ) )
            {
                this->paste_image();
            }
            if( ImGui::MenuItem( ICON_FA_PLUS "  Insert left", "ctrl " ICON_FA_LONG_ARROW_ALT_LEFT ) )
            {
                this->insert_image_left();
            }
            if( ImGui::MenuItem( ICON_FA_PLUS "  Insert right", "ctrl " ICON_FA_LONG_ARROW_ALT_RIGHT ) )
            {
                this->insert_image_right();
            }
            if( ImGui::MenuItem( ICON_FA_MINUS "  Delete system", "del" ) )
            {
                this->delete_image();
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Controls" ) )
        {
            previous_frame_menu_was_active = true;
            if( ImGui::MenuItem( "Start/stop calculation", "space" ) )
            {
                this->start_stop();
            }
            if( ImGui::MenuItem( "Randomize spins", "ctrl+r" ) )
            {
                Configuration_Random( state.get() );
                rendering_layer.needs_data();
                this->ui_shared_state.notify( "Randomized spins" );
            }
            if( ImGui::MenuItem( "Cycle method", "ctrl+m", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Cycle solver", "ctrl+s", false, false ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem(
                    "Toggle dragging mode", "F5",
                    ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG ) )
            {
                if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DRAG )
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );
                else
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::DRAG );
            }
            if( ImGui::MenuItem(
                    "Toggle defect mode", "F6",
                    ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT ) )
            {
                if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::DEFECT )
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );
                else
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::DEFECT );
            }
            if( ImGui::MenuItem(
                    "Toggle pinning mode", "F7",
                    ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING,
                    SPIRIT_PINNING_ENABLED ) )
            {
                if( ui_shared_state.interaction_mode == UiSharedState::InteractionMode::PINNING )
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );
                else
                    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::PINNING );
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Log" ) )
        {
            previous_frame_menu_was_active = true;
            if( ImGui::MenuItem( "System energy contributions", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "System magnetization", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "System topological charge", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Chain reaction coordinates", "", false, false ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "View" ) )
        {
            previous_frame_menu_was_active = true;
            ImGui::MenuItem( "Info-widgets", "i", &ui_config_file.show_overlays );
            ImGui::MenuItem( "Spin Configurations", "", &ui_config_file.show_configurations_widget );
            ImGui::MenuItem( "Parameters", "", &ui_config_file.show_parameters_widget );
            ImGui::MenuItem( "Plots", "F3", &ui_config_file.show_plots );
            ImGui::MenuItem( "Geometry", "", &ui_config_file.show_geometry_widget );
            ImGui::MenuItem( "Hamiltonian", "", &ui_config_file.show_hamiltonian_widget );
            ImGui::MenuItem( "Visualisation settings", "F4", &ui_config_file.show_visualisation_widget );
            ImGui::Separator();
            ImGui::MenuItem( "ImGui Demo Window", "", &show_imgui_demo_window );
            ImGui::MenuItem( "ImPlot Demo Window", "", &show_implot_demo_window );
            ImGui::Separator();
            if( ImGui::MenuItem( "Regular mode", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Isosurface mode", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode X", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode Y", "", false, false ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode Z", "", false, false ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem(
                    "Orthographic camera", "c", ui_shared_state.camera_is_orthographic,
                    ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR ) )
            {
                this->rendering_layer.set_camera_orthographic( !ui_shared_state.camera_is_orthographic );
            }
            if( ImGui::MenuItem(
                    "Reset camera", "ctrl+shift+r", false,
                    ui_shared_state.interaction_mode == UiSharedState::InteractionMode::REGULAR ) )
            {
                this->rendering_layer.reset_camera();
            }
            ImGui::Separator();
            // if( ImGui::MenuItem( "Toggle visualisation", "ctrl+shift+v", false, false ) )
            // {
            // }
            if( ImGui::MenuItem( "Hide menu bar", "ctrl+f", ui_config_file.window_hide_menubar ) )
            {
            }
            if( ImGui::MenuItem( "Fullscreen", "F11, ctrl+shift+f", ui_config_file.window_fullscreen ) )
            {
                fullscreen_toggled = true;
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Help" ) )
        {
            previous_frame_menu_was_active = true;
            if( ImGui::MenuItem( "Keybindings", "F1" ) )
            {
                show_keybindings = true;
            }
            if( ImGui::MenuItem( "About" ) )
            {
                show_about = true;
            }
            ImGui::EndMenu();
        }

        float menu_end = ImGui::GetCursorPosX();

        auto io             = ImGui::GetIO();
        auto & style        = ImGui::GetStyle();
        float right_edge    = ImGui::GetWindowContentRegionMax().x;
        float button_height = ImGui::GetWindowContentRegionMax().y + style.FramePadding.y;
        float bar_height    = ImGui::GetWindowContentRegionMax().y + 2 * style.FramePadding.y;
        float width, height;

        ImGui::PushStyleVar( ImGuiStyleVar_SelectableTextAlign, ImVec2( .5f, .5f ) );

        // TODO: deactivate method selection if a calculation is running

        for( int n = modes.size(); n > 0; n-- )
        {
            auto mode         = GUI_Mode( n );
            std::string label = modes[mode].first;
            ImVec2 text_size  = ImGui::CalcTextSize( label.c_str(), NULL, true );
            width             = text_size.x + 2 * style.FramePadding.x;
            height            = text_size.y + 2 * style.FramePadding.y;

            ImGui::SameLine( right_edge - width, 0 );
            if( ImGui::Selectable( label.c_str(), ui_shared_state.selected_mode == mode, 0, ImVec2( width, height ) ) )
                ui_shared_state.selected_mode = mode;
            right_edge -= ( width + 2 * style.FramePadding.x );

            if( ImGui::IsItemHovered() )
            {
                // 1.5s delay before showing tooltip
                mode_button_hovered_duration[n - 1] += io.DeltaTime;
                if( mode_button_hovered_duration[n - 1] > 1.5f )
                {
                    ImGui::BeginTooltip();
                    ImGui::TextUnformatted( modes[mode].second.c_str() );
                    ImGui::EndTooltip();
                }
            }
            else
            {
                mode_button_hovered_duration[n - 1] = 0;
            }
        }

        width             = 2.f * font_size_px;
        float total_width = 3 * width + 40 + 4 * style.FramePadding.x;
        float start       = menu_end + 0.5f * ( right_edge - menu_end - total_width );

        ImGui::SameLine( start, 0 );
        bool currently_running
            = Simulation_Running_On_Chain( state.get() ) || Simulation_Running_On_Image( state.get() );

        if( calculation_running && !currently_running )
        {
            calculation_running = false;
            this->ui_shared_state.notify( "calculation ended" );
        }

        if( currently_running )
        {
            calculation_running = true;
            if( ImGui::Button( ICON_FA_STOP, ImVec2( width, button_height ) ) )
            {
                this->start_stop();
                calculation_running = false;
            }
        }
        else
        {
            calculation_running = false;
            if( ImGui::Button( ICON_FA_PLAY, ImVec2( width, button_height ) ) )
            {
                this->start_stop();
                calculation_running = true;
            }
        }

        if( ImGui::Button( ICON_FA_ARROW_LEFT, ImVec2( width, button_height ) ) )
        {
            if( System_Get_Index( state.get() ) > 0 )
            {
                Chain_prev_Image( this->state.get() );
                rendering_layer.needs_data();
                parameters_widget.update_data();
            }
        }

        image_number = static_cast<ImU32>( System_Get_Index( state.get() ) + 1 );
        chain_length = static_cast<ImU32>( Chain_Get_NOI( state.get() ) );
        ImGui::SetNextItemWidth( 40 );
        if( ImGui::InputScalar(
                "##imagenumber", ImGuiDataType_U32, &image_number, NULL, NULL, "%u",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            Chain_Jump_To_Image( state.get(), image_number - 1 );
            rendering_layer.needs_data();
            parameters_widget.update_data();
        }
        if( ImGui::IsItemActive() )
            previous_frame_menu_was_active = true;
        ImGui::TextUnformatted( "/" );
        ImGui::SetNextItemWidth( 40 );
        if( ImGui::InputScalar(
                "##chainlength", ImGuiDataType_U32, &chain_length, NULL, NULL, "%u",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            const int length = Chain_Get_NOI( state.get() );
            Chain_Set_Length( state.get(), chain_length );
            const int new_length = Chain_Get_NOI( state.get() );
            // Resize the threads_image vector
            if( new_length < length )
            {
                for( int i = length - 1; i >= new_length; --i )
                    if( threads_image[i].joinable() )
                        threads_image[i].join();
                threads_image.pop_back();
            }
            else
            {
                this->threads_image.resize( new_length ); //( threads_image.begin() + idx + 1, std::thread() );
            }
        }
        if( ImGui::IsItemActive() )
            previous_frame_menu_was_active = true;

        if( ImGui::Button( ICON_FA_ARROW_RIGHT, ImVec2( width, button_height ) ) )
        {
            if( System_Get_Index( state.get() ) < Chain_Get_NOI( this->state.get() ) - 1 )
            {
                Chain_next_Image( this->state.get() );
                rendering_layer.needs_data();
                parameters_widget.update_data();
            }
        }

        ImGui::PopStyleVar(); // ImGuiStyleVar_SelectableTextAlign

        menu_bar_size = ImGui::GetWindowSize();
        ImGui::EndMainMenuBar();
    }
    // ImGui::PopFont();
}

void MainWindow::show_notifications()
{
    const ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                          | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                          | ImGuiWindowFlags_NoNav;

    const float fade_time = 0.25f; // seconds

    auto & io    = ImGui::GetIO();
    auto & style = ImGui::GetStyle();

    int i_notification = 0;
    float pos_y        = io.DisplaySize.y - 20;

    for( auto notification_iterator = ui_shared_state.notifications.begin();
         notification_iterator != ui_shared_state.notifications.end(); )
    {
        auto & notification = *notification_iterator;
        if( notification.timer > notification.timeout )
        {
            notification_iterator = ui_shared_state.notifications.erase( notification_iterator );
            continue;
        }

        float alpha = 0.8f;

        ImVec2 text_size = ImGui::CalcTextSize( notification.message.c_str() );
        float distance   = text_size.y + 2 * style.FramePadding.y + 2 * style.WindowPadding.y;
        pos_y -= distance;

        if( notification.timer < fade_time )
        {
            auto rad = 1.5707963f * notification.timer / fade_time;
            alpha *= std::sin( rad );
            pos_y += distance * std::cos( rad );
        }
        else if( notification.timer > notification.timeout - fade_time )
        {
            auto rad = 1.5707963f * ( notification.timeout - notification.timer ) / fade_time;
            alpha *= std::sin( rad );
            pos_y += distance * std::cos( rad );
        }

        alpha = std::max( 0.000001f, alpha );

        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0 );
        ImGui::PushStyleVar( ImGuiStyleVar_Alpha, alpha );

        ImGui::SetNextWindowPos( { 0.5f * ( io.DisplaySize.x - text_size.x ), pos_y } );
        // Also need to set size, because window may otherwise flicker for some reason...
        ImGui::SetNextWindowSize(
            { text_size.x + 2 * style.WindowPadding.x, text_size.y + 2 * style.WindowPadding.y } );

        if( ImGui::Begin( fmt::format( "Notification{}", i_notification ).c_str(), nullptr, window_flags ) )
        {
            ImGui::TextUnformatted( notification.message.c_str() );
            ImGui::End();
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleVar();

        notification.timer += io.DeltaTime;

        ++notification_iterator;
        ++i_notification;
    }
}

int MainWindow::run()
{
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop( emscripten_loop, 0, true );
#else
    while( !glfwWindowShouldClose( glfw_window ) )
    {
        glfwPollEvents();
        this->draw();
    }
#endif

    return 0;
}

MainWindow::MainWindow( std::shared_ptr<State> state )
        : GlfwWindow( "Spirit - Magnetism Simulation Tool" ),
          state( state ),
          rendering_layer( ui_shared_state, state ),
          ui_config_file( ui_shared_state, rendering_layer ),
          configurations_widget( ui_config_file.show_configurations_widget, state, ui_shared_state, rendering_layer ),
          parameters_widget( ui_config_file.show_parameters_widget, state, ui_shared_state ),
          geometry_widget( ui_config_file.show_geometry_widget, state, rendering_layer ),
          hamiltonian_widget( ui_config_file.show_hamiltonian_widget, state, rendering_layer ),
          plots_widget( ui_config_file.show_plots, state ),
          visualisation_widget( ui_config_file.show_visualisation_widget, state, rendering_layer )
{
    global_window_handle = this;

    this->threads_image = std::vector<std::thread>( Chain_Get_NOI( this->state.get() ) );

    gladLoadGL( (GLADloadfunc)glfwGetProcAddress ); // Initialize GLAD

    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO & io = ImGui::GetIO();

    // Enable docking
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // Disable split docking

    io.IniFilename = nullptr;

    auto & plot_style     = ImPlot::GetStyle();
    plot_style.FitPadding = { 0.08f, 0.08f };

    ImGui_ImplGlfw_InitForOpenGL( glfw_window, false );
    ImGui_ImplOpenGL3_Init();

#ifdef __EMSCRIPTEN__
    EmscriptenWebGLContextAttributes attrs_imgui;
    emscripten_webgl_init_context_attributes( &attrs_imgui );
    attrs_imgui.majorVersion = 1;
    attrs_imgui.minorVersion = 0;
    attrs_imgui.alpha        = 1;

    EmscriptenWebGLContextAttributes attrs_vfr;
    emscripten_webgl_init_context_attributes( &attrs_vfr );
    attrs_vfr.majorVersion = 1;
    attrs_vfr.minorVersion = 0;

    context_imgui = emscripten_webgl_create_context( "#imgui-canvas", &attrs_imgui );
    context_vfr   = emscripten_webgl_create_context( "#vfr-canvas", &attrs_vfr );

    ui_config_file.window_size[0]     = canvas_get_width();
    ui_config_file.window_size[1]     = canvas_get_height();
    ui_config_file.window_position[0] = 0;
    ui_config_file.window_position[1] = 0;

    emscripten_webgl_make_context_current( context_imgui );
#endif
    Log_Send(
        state.get(), Log_Level_Info, Log_Sender_UI,
        fmt::format( "GUI using OpenGL version {}", reinterpret_cast<const char *>( glGetString( GL_VERSION ) ) )
            .c_str() );

    glfwSetWindowPos( glfw_window, ui_config_file.window_position[0], ui_config_file.window_position[1] );
    glfwSetWindowSize( glfw_window, ui_config_file.window_size[0], ui_config_file.window_size[1] );
    if( ui_config_file.window_maximized )
        glfwMaximizeWindow( glfw_window );

    bool icon_set = images::glfw_set_app_icon( glfw_window );

    rendering_layer.initialize_gl();

    // Setup style
    if( ui_shared_state.dark_mode )
        styles::apply_charcoal();
    else
        styles::apply_light();

    // Load Fonts
    font_cousine_14 = fonts::cousine( 14 );
    font_karla_14   = fonts::karla( 14 );
    font_karla_16   = fonts::karla( 16 );

    glfwSetScrollCallback( glfw_window, ImGui_ImplGlfw_ScrollCallback );
    glfwSetMouseButtonCallback( glfw_window, ImGui_ImplGlfw_MouseButtonCallback );
    glfwSetKeyCallback( glfw_window, ImGui_ImplGlfw_KeyCallback );
    glfwSetCharCallback( glfw_window, ImGui_ImplGlfw_CharCallback );

    glfwSetFramebufferSizeCallback( glfw_window, framebuffer_size_callback );

#ifdef __EMSCRIPTEN__
    resizeCanvas();
#endif
}

MainWindow::~MainWindow()
{
    // Stop and wait for any running calculations
    this->stop_all();

    // Return to the regular mode
    this->rendering_layer.set_interaction_mode( UiSharedState::InteractionMode::REGULAR );

    // Update the config
    glfwGetWindowPos( glfw_window, &ui_config_file.window_position[0], &ui_config_file.window_position[1] );
    glfwGetWindowSize( glfw_window, &ui_config_file.window_size[0], &ui_config_file.window_size[1] );
    ui_config_file.window_maximized = glfwGetWindowAttrib( glfw_window, GLFW_MAXIMIZED );

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
}

void MainWindow::resize( int width, int height )
{
    rendering_layer.view.setFramebufferSize( width, height );
    rendering_layer.needs_redraw();
    this->draw();
}

} // namespace ui
