#include <imgui_impl/glfw.h>
#include <imgui_impl/opengl3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
// #include <GLES3/gl3.h>
#endif

#include <glad/glad.h>

#include <enums.hpp>
#include <fonts.hpp>
#include <main_window.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <GLFW/glfw3.h>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/View.hxx>

#include <imgui/imgui_internal.h>

#include <glm/gtc/type_ptr.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <fmt/format.h>

#include <exception>

static GLFWwindow * glfw_window;
static ImVec4 background_colour = ImVec4( 0.4f, 0.4f, 0.4f, 0.f );
static bool show_demo_window    = false;
static GUI_Mode selected_mode   = GUI_Mode::Minimizer;

static bool dark_mode = true;

static bool drag_main_window = false;
static double wx_start, wy_start;
static double cx_start, cy_start;
static bool main_window_maximized = false;

static ImFont * font_14 = nullptr;
static ImFont * font_16 = nullptr;
static ImFont * font_18 = nullptr;

static VFRendering::View vfr_view;
static VFRendering::Geometry vfr_geometry;
static VFRendering::VectorField vfr_vectorfield        = VFRendering::VectorField( {}, {} );
static VFRendering::VectorField vfr_vectorfield_surf2D = VFRendering::VectorField( {}, {} );
// static std::shared_ptr<VFRendering::ArrowRenderer> vfr_arrow_renderer_ptr;
static std::vector<std::shared_ptr<VFRendering::RendererBase>> vfr_renderers( 0 );
static int n_cell_step = 1;

static main_window * global_window_handle;

/////////////////////////////////////////////////////////////////////

#ifdef __EMSCRIPTEN__
EM_JS( int, canvas_get_width, (), { return Module.canvas.width; } );
EM_JS( int, canvas_get_height, (), { return Module.canvas.height; } );
EM_JS( void, resizeCanvas, (), { js_resizeCanvas(); } );

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_imgui;
EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_vfr;

void emscripten_loop()
{
    global_window_handle->loop();
}
#endif

static void glfw_error_callback( int error, const char * description )
{
    fmt::print( "Glfw Error {}: {}\n", error, description );
}

/////////////////////////////////////////////////////////////////////

void main_window::intitialize_gl()
{
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

    int width  = canvas_get_width();
    int height = canvas_get_height();

    emscripten_webgl_make_context_current( context_imgui );
    glfwSetWindowSize( glfw_window, width, height );
#endif
    fmt::print( "OpenGL Version: {}\n", glGetString( GL_VERSION ) );

    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( 0.125f );
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( 0.3f );
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( 0.0625f );
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( 0.35f );
    vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
        { background_colour.x, background_colour.y, background_colour.z } );

    this->update_vf_geometry();
    this->update_vf_directions();

    this->reset_camera();

    vfr_view.setOption<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(
        VFRendering::Utilities::getColormapImplementation( VFRendering::Utilities::Colormap::HSV ) );

    auto vfr_arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>( vfr_view, vfr_vectorfield );
    vfr_renderers.push_back( vfr_arrow_renderer_ptr );

    vfr_view.renderers(
        { { std::make_shared<VFRendering::CombinedRenderer>( vfr_view, vfr_renderers ), { { 0, 0, 1, 1 } } } } );
}

void main_window::reset_camera()
{
    float camera_distance = 30.0f;
    // auto center_position  = ( vfr_geometry.min() + vfr_geometry.max() ) * 0.5f;
    float b_min[3], b_max[3];
    Geometry_Get_Bounds( state.get(), b_min, b_max );
    glm::vec3 bounds_min      = glm::make_vec3( b_min );
    glm::vec3 bounds_max      = glm::make_vec3( b_max );
    glm::vec3 center_position = ( bounds_min + bounds_max ) * 0.5f;
    auto camera_position      = center_position + camera_distance * glm::vec3( 0, 0, 1 );
    auto up_vector            = glm::vec3( 0, 1, 0 );

    VFRendering::Options options;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>( center_position );
    // options.set<VFRendering::View::Option::SYSTEM_CENTER>( { 0, 0, 0 } );
    options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( 45 );
    options.set<VFRendering::View::Option::CAMERA_POSITION>( camera_position );
    options.set<VFRendering::View::Option::CENTER_POSITION>( center_position );
    options.set<VFRendering::View::Option::UP_VECTOR>( up_vector );
    vfr_view.updateOptions( options );

    fmt::print( "min       {} {} {}\n", vfr_geometry.min().x, vfr_geometry.min().y, vfr_geometry.min().z );
    fmt::print( "max       {} {} {}\n", vfr_geometry.max().x, vfr_geometry.max().y, vfr_geometry.max().z );
    auto sys_center = options.get<VFRendering::View::Option::SYSTEM_CENTER>();
    fmt::print( "system center at {} {} {}\n", sys_center.x, sys_center.y, sys_center.z );
    auto cam_center = options.get<VFRendering::View::Option::CENTER_POSITION>();
    fmt::print( "camera center at {} {} {}\n", cam_center.x, cam_center.y, cam_center.z );
    auto cam = options.get<VFRendering::View::Option::CAMERA_POSITION>();
    fmt::print( "camera position at {} {} {}\n", cam.x, cam.y, cam.z );

    // needs_redraw = true;
}

void main_window::handle_mouse()
{
    auto & io = ImGui::GetIO();

#if defined( __APPLE__ )
    float scroll = -0.1 * io.MouseWheel;
#elif defined( __EMSCRIPTEN__ )
    float scroll = ( io.MouseWheel > 0 ? 1 : ( io.MouseWheel < 0 ? -1 : 0 ) );
#else
    float scroll = -io.MouseWheel;
#endif

    if( !io.KeyShift )
        scroll *= 10;

    if( io.MouseWheel )
    {
        vfr_view.mouseScroll( scroll );
    }

    float scale = 1;
    if( io.KeyShift )
        scale = 0.1f;

    if( ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT ) )
    {
        vfr_view.mouseMove(
            glm::vec2( 0, 0 ), glm::vec2( scale * io.MouseDelta.x, scale * io.MouseDelta.y ),
            VFRendering::CameraMovementModes::ROTATE_BOUNDED );
    }
    else if( ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) )
    {
        vfr_view.mouseMove(
            glm::vec2( 0, 0 ), glm::vec2( scale * io.MouseDelta.x, scale * io.MouseDelta.y ),
            VFRendering::CameraMovementModes::TRANSLATE );
    }
}

void main_window::handle_keyboard()
{
    auto & io = ImGui::GetIO();

#ifdef __APPLE__
    bool ctrl = io.KeySuper;
#else
    bool ctrl    = io.KeyCtrl;
#endif

    if( ctrl && io.KeyShift )
    {
        if( ImGui::IsKeyPressed( GLFW_KEY_R ) )
        {
            this->reset_camera();
        }
    }
    else if( ctrl )
    {
        if( ImGui::IsKeyPressed( GLFW_KEY_R ) )
        {
            Configuration_Random( state.get() );
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( GLFW_KEY_X ) )
        {
            if( Chain_Get_NOI( state.get() ) > 1 )
            {
                stop_current();

                Chain_Image_to_Clipboard( state.get() );

                int idx = System_Get_Index( state.get() );
                if( Chain_Delete_Image( state.get(), idx ) )
                {
                    // Make the llg_threads vector smaller
                    if( this->threads_image[idx].joinable() )
                        this->threads_image[idx].join();
                    this->threads_image.erase( threads_image.begin() + idx );
                }
            }
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_C ) )
        {
            Chain_Image_to_Clipboard( global_window_handle->state.get() );
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_V ) )
        {
            // Paste a Spin System into current System
            this->stop_current();
            Chain_Replace_Image( state.get() );
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_LEFT ) )
        {
            int idx = System_Get_Index( state.get() );
            // Insert Image
            Chain_Insert_Image_Before( state.get() );
            // Make the llg_threads vector larger
            this->threads_image.insert( threads_image.begin() + idx, std::thread() );
            // Switch to the inserted image
            Chain_prev_Image( this->state.get() );
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_RIGHT ) )
        {
            int idx = System_Get_Index( state.get() );
            // Insert Image
            Chain_Insert_Image_After( state.get() );
            // Make the llg_threads vector larger
            this->threads_image.insert( threads_image.begin() + idx + 1, std::thread() );
            // Switch to the inserted image
            Chain_next_Image( this->state.get() );
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

        if( ImGui::IsKeyPressed( GLFW_KEY_W ) && !ImGui::IsKeyPressed( GLFW_KEY_S ) )
        {
            vfr_view.mouseScroll( -scale );
        }
        else if( ImGui::IsKeyPressed( GLFW_KEY_S ) && !ImGui::IsKeyPressed( GLFW_KEY_W ) )
        {
            vfr_view.mouseScroll( scale );
        }

        bool rotate_camera = false;
        bool move_camera   = false;
        float dx           = 0;
        float dy           = 0;
        float theta        = 0;
        float phi          = 0;

        if( ImGui::IsKeyPressed( GLFW_KEY_A ) && !ImGui::IsKeyPressed( GLFW_KEY_D ) )
        {
            rotate_camera = true;
            phi           = 5 * scale;
        }
        else if( ImGui::IsKeyPressed( GLFW_KEY_D ) && !ImGui::IsKeyPressed( GLFW_KEY_A ) )
        {
            rotate_camera = true;
            phi           = -5 * scale;
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_Q ) && !ImGui::IsKeyPressed( GLFW_KEY_E ) )
        {
            rotate_camera = true;
            theta         = 5 * scale;
        }
        else if( ImGui::IsKeyPressed( GLFW_KEY_E ) && !ImGui::IsKeyPressed( GLFW_KEY_Q ) )
        {
            rotate_camera = true;
            theta         = -5 * scale;
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_T ) && !ImGui::IsKeyPressed( GLFW_KEY_G ) )
        {
            move_camera = true;
            dy          = scale;
        }
        else if( ImGui::IsKeyPressed( GLFW_KEY_G ) && !ImGui::IsKeyPressed( GLFW_KEY_T ) )
        {
            move_camera = true;
            dy          = -scale;
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_F ) && !ImGui::IsKeyPressed( GLFW_KEY_H ) )
        {
            move_camera    = true;
            float duration = io.KeysDownDuration[GLFW_KEY_T];
            dx             = scale;
        }
        else if( ImGui::IsKeyPressed( GLFW_KEY_H ) && !ImGui::IsKeyPressed( GLFW_KEY_F ) )
        {
            move_camera = true;
            dx          = -scale;
        }

        if( rotate_camera )
            vfr_view.mouseMove( { 0, 0 }, { phi, theta }, VFRendering::CameraMovementModes::ROTATE_BOUNDED );
        if( move_camera )
            vfr_view.mouseMove( { 0, 0 }, { dx, dy }, VFRendering::CameraMovementModes::TRANSLATE );

        // Reset the key repeat parameters
        io.KeyRepeatRate  = backup_repeat_rate;
        io.KeyRepeatDelay = backup_repeat_delay;

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( GLFW_KEY_X, false ) )
        {
            float camera_distance = glm::length(
                vfr_view.options().get<VFRendering::View::Option::CENTER_POSITION>()
                - vfr_view.options().get<VFRendering::View::Option::CAMERA_POSITION>() );
            auto center_position = vfr_view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
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
            vfr_view.updateOptions( options );
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_Y, false ) )
        {
            float camera_distance = glm::length(
                vfr_view.options().get<VFRendering::View::Option::CENTER_POSITION>()
                - vfr_view.options().get<VFRendering::View::Option::CAMERA_POSITION>() );
            auto center_position = vfr_view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
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
            vfr_view.updateOptions( options );
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_Z, false ) )
        {
            float camera_distance = glm::length(
                vfr_view.options().get<VFRendering::View::Option::CENTER_POSITION>()
                - vfr_view.options().get<VFRendering::View::Option::CAMERA_POSITION>() );
            auto center_position = vfr_view.options().get<VFRendering::View::Option::SYSTEM_CENTER>();
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
            vfr_view.updateOptions( options );
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( GLFW_KEY_SPACE, false ) )
        {
            start_stop();
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_RIGHT ) )
        {
            if( System_Get_Index( state.get() ) < Chain_Get_NOI( this->state.get() ) - 1 )
            {
                // Change active image
                Chain_next_Image( this->state.get() );

                // // Update
                // this->updateData();
            }
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_LEFT ) )
        {
            // this->return_focus();
            if( System_Get_Index( state.get() ) > 0 )
            {
                // Change active image!
                Chain_prev_Image( this->state.get() );

                // // Update
                // this->updateData();
            }
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_DELETE ) )
        {
            if( Chain_Get_NOI( state.get() ) > 1 )
            {
                this->stop_current();

                int idx = System_Get_Index( state.get() );
                if( Chain_Delete_Image( state.get() ) )
                {
                    // Make the llg_threads vector smaller
                    if( this->threads_image[idx].joinable() )
                        this->threads_image[idx].join();
                    this->threads_image.erase( threads_image.begin() + idx );
                }

                //     // Update
                //     this->updateData();
            }
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( GLFW_KEY_F1, false ) )
        {
            show_keybindings = !show_keybindings;
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_I, false ) )
        {
            show_overlays = !show_overlays;
        }

        //-----------------------------------------------------
        // TODO: deactivate method selection if a calculation is running
        if( ImGui::IsKeyPressed( GLFW_KEY_1, false ) )
        {
            selected_mode = GUI_Mode::Minimizer;
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_2, false ) )
        {
            selected_mode = GUI_Mode::MC;
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_3, false ) )
        {
            selected_mode = GUI_Mode::LLG;
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_4, false ) )
        {
            selected_mode = GUI_Mode::GNEB;
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_5, false ) )
        {
            selected_mode = GUI_Mode::MMF;
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_6, false ) )
        {
            selected_mode = GUI_Mode::EMA;
        }
    }
}

void main_window::start_stop() try
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
    }
    else
    {
        // Not running, so we start it
        if( selected_mode == GUI_Mode::Minimizer )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_LLG_Start, this->state.get(), selected_solver, -1, -1, false, -1, -1 );
        }
        if( selected_mode == GUI_Mode::LLG )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_LLG_Start, this->state.get(), selected_solver, -1, -1, false, -1, -1 );
        }
        else if( selected_mode == GUI_Mode::MC )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_MC_Start, this->state.get(), -1, -1, false, -1, -1 );
        }
        else if( selected_mode == GUI_Mode::GNEB )
        {
            if( thread_chain.joinable() )
                thread_chain.join();
            this->thread_chain
                = std::thread( &Simulation_GNEB_Start, this->state.get(), selected_solver, -1, -1, false, -1 );
        }
        else if( selected_mode == GUI_Mode::MMF )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_MMF_Start, this->state.get(), selected_solver, -1, -1, false, -1, -1 );
        }
        else if( selected_mode == GUI_Mode::EMA )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_EMA_Start, this->state.get(), -1, -1, false, -1, -1 );
        }
    }
}
catch( const std::exception & e )
{
    Log_Send(
        state.get(), Log_Level_Error, Log_Sender_UI, fmt::format( "caught std::exception: {}\n", e.what() ).c_str() );
}

void main_window::stop_all() try
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
}
catch( const std::exception & e )
{
    Log_Send(
        state.get(), Log_Level_Error, Log_Sender_UI, fmt::format( "caught std::exception: {}\n", e.what() ).c_str() );
}

void main_window::stop_current() try
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
}
catch( const std::exception & e )
{
    Log_Send(
        state.get(), Log_Level_Error, Log_Sender_UI, fmt::format( "caught std::exception: {}\n", e.what() ).c_str() );
}

void main_window::loop()
{
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

    glfwPollEvents();

    auto & io = ImGui::GetIO();

    if( !io.WantCaptureMouse )
        this->handle_mouse();

    if( !io.WantCaptureKeyboard )
        this->handle_keyboard();

    draw_imgui( display_w, display_h );

    glfwSwapBuffers( glfw_window );

#ifdef __EMSCRIPTEN__
    emscripten_webgl_make_context_current( context_vfr );
#endif

    draw_vfr( display_w, display_h );
}

void main_window::draw_vfr( int display_w, int display_h )
{
    update_vf_directions();
    vfr_view.setFramebufferSize( float( display_w ), float( display_h ) );
    vfr_view.draw();
}

void main_window::update_vf_geometry()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, n_cells[0] / n_cell_step ), std::max( 1, n_cells[1] / n_cell_step ),
                            std::max( 1, n_cells[2] / n_cell_step ) };
    int nos_draw        = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Positions of the vectorfield
    std::vector<glm::vec3> positions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Positions
    //        get pointer
    scalar * spin_pos;
    int * atom_types;
    spin_pos   = Geometry_Get_Positions( state.get() );
    atom_types = Geometry_Get_Atom_Types( state.get() );
    int icell  = 0;
    for( int cell_c = 0; cell_c < n_cells_draw[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells_draw[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells_draw[0]; cell_a++ )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis
                              + n_cell_atoms * n_cell_step
                                    * ( +cell_a + n_cells[0] * cell_b + n_cells[0] * n_cells[1] * cell_c );
                    positions[icell] = glm::vec3( spin_pos[3 * idx], spin_pos[1 + 3 * idx], spin_pos[2 + 3 * idx] );
                    ++icell;
                }
            }
        }
    }

    // Generate the right geometry (triangles and tetrahedra)
    VFRendering::Geometry geometry;
    VFRendering::Geometry geometry_surf2D;
    //      get tetrahedra
    if( Geometry_Get_Dimensionality( state.get() ) == 3 )
    {
        if( n_cell_step > 1
            && ( n_cells[0] / n_cell_step < 2 || n_cells[1] / n_cell_step < 2 || n_cells[2] / n_cell_step < 2 ) )
        {
            geometry = VFRendering::Geometry( positions, {}, {}, true );
        }
        else
        {
            const std::array<VFRendering::Geometry::index_type, 4> * tetrahedra_indices_ptr = nullptr;
            int num_tetrahedra                                                              = Geometry_Get_Tetrahedra(
                state.get(), reinterpret_cast<const int **>( &tetrahedra_indices_ptr ), n_cell_step );
            std::vector<std::array<VFRendering::Geometry::index_type, 4>> tetrahedra_indices(
                tetrahedra_indices_ptr, tetrahedra_indices_ptr + num_tetrahedra );
            geometry = VFRendering::Geometry( positions, {}, tetrahedra_indices, false );
        }
    }
    else if( Geometry_Get_Dimensionality( state.get() ) == 2 )
    {
        // Determine two basis vectors
        std::array<glm::vec3, 2> basis;
        float eps = 1e-6;
        for( int i = 1, j = 0; i < nos && j < 2; ++i )
        {
            if( glm::length( positions[i] - positions[0] ) > eps )
            {
                if( j < 1 )
                {
                    basis[j] = glm::normalize( positions[i] - positions[0] );
                    ++j;
                }
                else
                {
                    if( 1 - std::abs( glm::dot( basis[0], glm::normalize( positions[i] - positions[0] ) ) ) > eps )
                    {
                        basis[j] = glm::normalize( positions[i] - positions[0] );
                        ++j;
                    }
                }
            }
        }
        glm::vec3 normal = glm::normalize( glm::cross( basis[0], basis[1] ) );
        // By default, +z is up, which is where we want the normal oriented towards
        if( glm::dot( normal, glm::vec3{ 0, 0, 1 } ) < 1e-6 )
            normal = -normal;

        // Rectilinear with one basis atom
        if( n_cell_atoms == 1 && std::abs( glm::dot( basis[0], basis[1] ) ) < 1e-6 )
        {
            std::vector<float> xs( n_cells_draw[0] ), ys( n_cells_draw[1] ), zs( n_cells_draw[2] );
            for( int i = 0; i < n_cells_draw[0]; ++i )
                xs[i] = positions[i].x;
            for( int i = 0; i < n_cells_draw[1]; ++i )
                ys[i] = positions[i * n_cells_draw[0]].y;
            for( int i = 0; i < n_cells_draw[2]; ++i )
                zs[i] = positions[i * n_cells_draw[0] * n_cells_draw[1]].z;
            geometry = VFRendering::Geometry::rectilinearGeometry( xs, ys, zs );
            for( int i = 0; i < n_cells_draw[0]; ++i )
                xs[i] = ( positions[i] - normal ).x;
            for( int i = 0; i < n_cells_draw[1]; ++i )
                ys[i] = ( positions[i * n_cells_draw[0]] - normal ).y;
            for( int i = 0; i < n_cells_draw[2]; ++i )
                zs[i] = ( positions[i * n_cells_draw[0] * n_cells_draw[1]] - normal ).z;
            geometry_surf2D = VFRendering::Geometry::rectilinearGeometry( xs, ys, zs );
        }
        // All others
        else
        {
            const std::array<VFRendering::Geometry::index_type, 3> * triangle_indices_ptr = nullptr;
            int num_triangles                                                             = Geometry_Get_Triangulation(
                state.get(), reinterpret_cast<const int **>( &triangle_indices_ptr ), n_cell_step );
            std::vector<std::array<VFRendering::Geometry::index_type, 3>> triangle_indices(
                triangle_indices_ptr, triangle_indices_ptr + num_triangles );
            geometry = VFRendering::Geometry( positions, triangle_indices, {}, true );
            for( int i = 0; i < nos_draw; ++i )
                positions[i] = positions[i] - normal;
            geometry_surf2D = VFRendering::Geometry( positions, triangle_indices, {}, true );
        }

        // Update the vectorfield geometry
        vfr_vectorfield_surf2D.updateGeometry( geometry_surf2D );
    }
    else
    {
        geometry = VFRendering::Geometry( positions, {}, {}, true );
    }

    // Update the vectorfield
    vfr_vectorfield.updateGeometry( geometry );
}

void main_window::update_vf_directions()
{
    int nos = System_Get_NOS( state.get() );
    int n_cells[3];
    Geometry_Get_N_Cells( this->state.get(), n_cells );
    int n_cell_atoms = Geometry_Get_N_Cell_Atoms( this->state.get() );

    int n_cells_draw[3] = { std::max( 1, n_cells[0] / n_cell_step ), std::max( 1, n_cells[1] / n_cell_step ),
                            std::max( 1, n_cells[2] / n_cell_step ) };
    int nos_draw        = n_cell_atoms * n_cells_draw[0] * n_cells_draw[1] * n_cells_draw[2];

    // Directions of the vectorfield
    std::vector<glm::vec3> directions = std::vector<glm::vec3>( nos_draw );

    // ToDo: Update the pointer to our Data instead of copying Data?
    // Directions
    //        get pointer
    scalar * spins;
    int * atom_types;
    atom_types = Geometry_Get_Atom_Types( state.get() );
    // if( this->m_source == 0 )
    spins = System_Get_Spin_Directions( state.get() );
    // else if( this->m_source == 1 )
    //     spins = System_Get_Effective_Field( state.get() );
    // else spins = System_Get_Spin_Directions( state.get() );

    //        copy
    /*positions.assign(spin_pos, spin_pos + 3*nos);
    directions.assign(spins, spins + 3*nos);*/
    int icell = 0;
    for( int cell_c = 0; cell_c < n_cells_draw[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells_draw[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells_draw[0]; cell_a++ )
            {
                for( int ibasis = 0; ibasis < n_cell_atoms; ++ibasis )
                {
                    int idx = ibasis + n_cell_atoms * cell_a * n_cell_step
                              + n_cell_atoms * n_cells[0] * cell_b * n_cell_step
                              + n_cell_atoms * n_cells[0] * n_cells[1] * cell_c * n_cell_step;
                    // std::cerr << idx << " " << icell << std::endl;
                    directions[icell] = glm::vec3( spins[3 * idx], spins[1 + 3 * idx], spins[2 + 3 * idx] );
                    if( atom_types[idx] < 0 )
                        directions[icell] *= 0;
                    ++icell;
                }
            }
        }
    }
    // //        rescale if effective field
    // if( this->m_source == 1 )
    // {
    //     float max_length = 0;
    //     for( auto direction : directions )
    //     {
    //         max_length = std::max( max_length, glm::length( direction ) );
    //     }
    //     if( max_length > 0 )
    //     {
    //         for( auto & direction : directions )
    //         {
    //             direction /= max_length;
    //         }
    //     }
    // }

    // Update the vectorfield
    vfr_vectorfield.updateVectors( directions );

    if( Geometry_Get_Dimensionality( state.get() ) == 2 )
        vfr_vectorfield_surf2D.updateVectors( directions );
}

void main_window::draw_imgui( int display_w, int display_h )
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::PushFont( font_14 );

    widgets::show_menu_bar(
        glfw_window, font_16, dark_mode, background_colour, selected_mode, selected_solver, vfr_view, show_keybindings,
        show_overlays, show_about, state, threads_image, thread_chain );

    widgets::show_overlay_system( show_overlays );
    widgets::show_overlay_calculation( show_overlays, selected_mode, selected_solver );

    widgets::show_parameters( selected_mode, show_parameters_settings );

    widgets::show_visualisation_settings( vfr_view, background_colour );

    widgets::show_energy_plot();

    widgets::show_keybindings( show_keybindings );

    widgets::show_about( show_about );

    if( show_demo_window )
    {
        ImGui::SetNextWindowPos( ImVec2( 650, 20 ), ImGuiCond_FirstUseEver );
        ImGui::ShowDemoWindow( &show_demo_window );
    }

    ImGui::Begin( "Test-window" );

    static float f     = 0.0f;
    static int counter = 0;
    ImGui::Text( "Hello, world!" );
    ImGui::SliderFloat( "float", &f, 0.0f, 1.0f );

    ImGui::Text( "Windows" );
    ImGui::Checkbox( "Demo Window", &show_demo_window );
    ImGui::Checkbox( "Keybindings", &show_keybindings );

    if( ImGui::Button( "Button" ) )
        counter++;
    ImGui::SameLine();
    ImGui::Text( "counter = %d", counter );

    ImGui::Text(
        "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate );

    ImGui::End();

    ImGui::PopFont();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );
}

void main_window::quit()
{
    // Stop and wait for any running calculations
    this->stop_all();

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow( glfw_window );
    glfwTerminate();
}

int main_window::run()
{
#ifdef __EMSCRIPTEN__
    emscripten_set_main_loop( emscripten_loop, 0, true );
#else
    while( !glfwWindowShouldClose( glfw_window ) )
    {
        loop();
    }
#endif

    quit();

    return 0;
}

main_window::main_window( std::shared_ptr<State> state )
{
    global_window_handle = this;

    this->state           = state;
    this->selected_solver = Solver_VP_OSO;
    this->threads_image   = std::vector<std::thread>( Chain_Get_NOI( this->state.get() ) );

    glfwSetErrorCallback( glfw_error_callback );

    if( !glfwInit() )
    {
        fmt::print( "Failed to initialize GLFW\n" );
        // return 1;
        throw std::runtime_error( "Failed to initialize GLFW" );
    }

    glfwWindowHint( GLFW_SAMPLES, 16 ); // 16x antialiasing
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 2 );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE ); // We don't want the old OpenGL
    // glfwWindowHint( GLFW_DECORATED, false );
    // glfwWindowHint( GLFW_RESIZABLE, true );
#if __APPLE__
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
#endif

    // Open a window and create its OpenGL context
    int canvasWidth  = 1280;
    int canvasHeight = 720;
    glfw_window      = glfwCreateWindow( canvasWidth, canvasHeight, "Spirit - Magnetism Simulation Tool", NULL, NULL );
    glfwMakeContextCurrent( glfw_window );
#ifndef __EMSCRIPTEN__
    glfwSwapInterval( 1 ); // Enable vsync
#endif

    if( glfw_window == NULL )
    {
        fmt::print( "Failed to open GLFW window.\n" );
        glfwTerminate();
        // return -1;
        throw std::runtime_error( "Failed to open GLFW window." );
    }

    gladLoadGL( (GLADloadfunc)glfwGetProcAddress ); // Initialize GLAD

    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io   = ImGui::GetIO();
    io.IniFilename = "imgui_state.ini";

    ImGui_ImplGlfw_InitForOpenGL( glfw_window, false );
    ImGui_ImplOpenGL3_Init();

    intitialize_gl();

    // Setup style
    styles::apply_charcoal();

    // Load Fonts
    font_14 = fonts::font_combined( 14 );
    font_16 = fonts::font_combined( 16 );
    font_18 = fonts::font_combined( 18 );

    glfwSetScrollCallback( glfw_window, ImGui_ImplGlfw_ScrollCallback );
    glfwSetMouseButtonCallback( glfw_window, ImGui_ImplGlfw_MouseButtonCallback );
    glfwSetKeyCallback( glfw_window, ImGui_ImplGlfw_KeyCallback );
    glfwSetCharCallback( glfw_window, ImGui_ImplGlfw_CharCallback );

#ifdef __EMSCRIPTEN__
    resizeCanvas();
#endif
}