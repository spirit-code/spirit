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

#include <imgui/imgui_internal.h>

#include <glm/gtc/type_ptr.hpp>

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/Geometry.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>

#include <fmt/format.h>

#include <nfd.h>

#include <exception>
#include <map>
#include <string>

static MainWindow * global_window_handle;

/////////////////////////////////////////////////////////////////////

#ifdef __EMSCRIPTEN__
EM_JS( int, canvas_get_width, (), { return Module.canvas.width; } );
EM_JS( int, canvas_get_height, (), { return Module.canvas.height; } );
EM_JS( void, resizeCanvas, (), { js_resizeCanvas(); } );

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_imgui;
EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_vfr;

void emscripten_loop()
{
    glfwPollEvents();
    global_window_handle->draw();
}
#endif

static void glfw_error_callback( int error, const char * description )
{
    fmt::print( "Glfw Error {}: {}\n", error, description );
}

static void framebufferSizeCallback( GLFWwindow * window, int width, int height )
{
    (void)window;
    global_window_handle->resize( width, height );
}

/////////////////////////////////////////////////////////////////////

void MainWindow::reset_camera()
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
    rendering_layer.view.updateOptions( options );

    rendering_layer.needs_redraw();
}

void MainWindow::handle_mouse()
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
        rendering_layer.view.mouseScroll( scroll );
        rendering_layer.needs_redraw();
    }

    float scale = 1;
    if( io.KeyShift )
        scale = 0.1f;

    if( ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT ) )
    {
        rendering_layer.view.mouseMove(
            glm::vec2( 0, 0 ), glm::vec2( scale * io.MouseDelta.x, scale * io.MouseDelta.y ),
            VFRendering::CameraMovementModes::ROTATE_BOUNDED );
        rendering_layer.needs_redraw();
    }
    else if( ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_RIGHT ) && !ImGui::IsMouseDragging( GLFW_MOUSE_BUTTON_LEFT ) )
    {
        rendering_layer.view.mouseMove(
            glm::vec2( 0, 0 ), glm::vec2( scale * io.MouseDelta.x, scale * io.MouseDelta.y ),
            VFRendering::CameraMovementModes::TRANSLATE );
        rendering_layer.needs_redraw();
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
        if( ImGui::IsKeyPressed( GLFW_KEY_R ) )
        {
            this->reset_camera();
            rendering_layer.needs_redraw();
        }
    }
    else if( ctrl )
    {
        if( ImGui::IsKeyPressed( GLFW_KEY_R ) )
        {
            Configuration_Random( state.get() );
            rendering_layer.needs_data();
        }

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( GLFW_KEY_X ) )
        {
            this->cut_image();
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_C ) )
        {
            Chain_Image_to_Clipboard( global_window_handle->state.get() );
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_V ) )
        {
            this->paste_image();
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_LEFT ) )
        {
            this->insert_image_left();
        }
        if( ImGui::IsKeyPressed( GLFW_KEY_RIGHT ) )
        {
            this->insert_image_right();
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
            rendering_layer.view.mouseScroll( -scale );
            rendering_layer.needs_redraw();
        }
        else if( ImGui::IsKeyPressed( GLFW_KEY_S ) && !ImGui::IsKeyPressed( GLFW_KEY_W ) )
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

        //-----------------------------------------------------

        if( ImGui::IsKeyPressed( GLFW_KEY_X, false ) )
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
        if( ImGui::IsKeyPressed( GLFW_KEY_Y, false ) )
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
        if( ImGui::IsKeyPressed( GLFW_KEY_Z, false ) )
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

                rendering_layer.needs_data();
            }
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_LEFT ) )
        {
            // this->return_focus();
            if( System_Get_Index( state.get() ) > 0 )
            {
                // Change active image!
                Chain_prev_Image( this->state.get() );

                rendering_layer.needs_data();
            }
        }

        if( ImGui::IsKeyPressed( GLFW_KEY_DELETE ) )
        {
            this->delete_image();
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
                = std::thread( &Simulation_LLG_Start, this->state.get(), selected_solver_min, -1, -1, false, -1, -1 );
        }
        if( selected_mode == GUI_Mode::LLG )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_LLG_Start, this->state.get(), selected_solver_llg, -1, -1, false, -1, -1 );
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
                = std::thread( &Simulation_GNEB_Start, this->state.get(), selected_solver_min, -1, -1, false, -1 );
        }
        else if( selected_mode == GUI_Mode::MMF )
        {
            int idx = System_Get_Index( state.get() );
            if( threads_image[idx].joinable() )
                threads_image[System_Get_Index( state.get() )].join();
            this->threads_image[System_Get_Index( state.get() )]
                = std::thread( &Simulation_MMF_Start, this->state.get(), selected_solver_min, -1, -1, false, -1, -1 );
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
}

void MainWindow::insert_image_left()
{
    if( Simulation_Running_On_Chain( state.get() ) )
        this->stop_current();

    int idx = System_Get_Index( state.get() );
    // Insert Image
    Chain_Insert_Image_Before( state.get() );
    // Make the llg_threads vector larger
    this->threads_image.insert( threads_image.begin() + idx, std::thread() );
    // Switch to the inserted image
    Chain_prev_Image( this->state.get() );
}

void MainWindow::insert_image_right()
{
    if( Simulation_Running_On_Chain( state.get() ) )
        this->stop_current();

    int idx = System_Get_Index( state.get() );
    // Insert Image
    Chain_Insert_Image_After( state.get() );
    // Make the llg_threads vector larger
    this->threads_image.insert( threads_image.begin() + idx + 1, std::thread() );
    // Switch to the inserted image
    Chain_next_Image( this->state.get() );
}

void MainWindow::delete_image()
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

        rendering_layer.needs_data();
    }
}

void MainWindow::draw()
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

    if( Simulation_Running_On_Image( this->state.get() ) || Simulation_Running_On_Chain( this->state.get() )
        || this->m_dragging )
    {
        rendering_layer.needs_data();
    }

    rendering_layer.draw( display_w, display_h );
}

void MainWindow::draw_imgui( int display_w, int display_h )
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::PushFont( font_karla_14 );

    this->show_menu_bar();

    ImGui::PushFont( font_cousine_14 );
    widgets::show_overlay_system( show_overlays );
    widgets::show_overlay_calculation( show_overlays, selected_mode, selected_solver_min, selected_solver_llg );
    ImGui::PopFont();

    widgets::show_parameters( show_parameters_settings, selected_mode );

    widgets::show_visualisation_settings( show_visualisation_settings, rendering_layer );

    widgets::show_plots( show_plots );

    widgets::show_keybindings( show_keybindings );

    widgets::show_about( show_about );

    if( show_demo_window )
    {
        ImGui::SetNextWindowPos( ImVec2( 650, 20 ), ImGuiCond_FirstUseEver );
        ImGui::ShowDemoWindow( &show_demo_window );
    }

    ImGui::Begin( "Test-window" );

    ImGui::Text( "Windows" );
    ImGui::Checkbox( "Demo Window", &show_demo_window );
    ImGui::Checkbox( "Keybindings", &show_keybindings );

    ImGui::Text(
        "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate );

    ImGui::End();

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
    static ImU32 image_number = (ImU32)1;
    static ImU32 chain_length = (ImU32)1;

    ImGui::PushFont( font_karla_16 );
    float font_size_px = font_karla_16->FontSize;

    ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 7.f, 7.f ) );
    if( ImGui::BeginMainMenuBar() )
    {
        ImGui::PopStyleVar();

        if( ImGui::BeginMenu( "File" ) )
        {
            if( ImGui::MenuItem( "Load config-file" ) )
            {
                nfdpathset_t pathSet;
                nfdresult_t result = NFD_OpenDialogMultiple( "cfg", NULL, &pathSet );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &pathSet ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &pathSet, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &pathSet );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save current cfg" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "cfg", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Load spin configuration" ) )
            {
                nfdpathset_t pathSet;
                nfdresult_t result = NFD_OpenDialogMultiple( "ovf;txt;csv", NULL, &pathSet );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &pathSet ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &pathSet, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &pathSet );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save spin configuration" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "ovf;txt;csv", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Load system eigenmodes" ) )
            {
            }
            if( ImGui::MenuItem( "Save system eigenmodes" ) )
            {
            }
            if( ImGui::MenuItem( "Save energy per spin" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Load chain" ) )
            {
                nfdpathset_t pathSet;
                nfdresult_t result = NFD_OpenDialogMultiple( "ovf;txt;csv", NULL, &pathSet );
                if( result == NFD_OKAY )
                {
                    size_t i;
                    for( i = 0; i < NFD_PathSet_GetCount( &pathSet ); ++i )
                    {
                        nfdchar_t * path = NFD_PathSet_GetPath( &pathSet, i );
                        fmt::print( "File open path {}: \"{}\"\n", (int)i, path );
                    }
                    NFD_PathSet_Free( &pathSet );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save chain" ) )
            {
                nfdchar_t * savePath = NULL;
                nfdresult_t result   = NFD_SaveDialog( "ovf;txt;csv", NULL, &savePath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "File save path: \"{}\"\n", savePath );
                    free( savePath );
                }
                else if( result != NFD_CANCEL )
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            if( ImGui::MenuItem( "Save energies" ) )
            {
            }
            if( ImGui::MenuItem( "Save interpolated energies" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Choose output folder" ) )
            {
                nfdchar_t * outPath = NULL;
                nfdresult_t result  = NFD_PickFolder( NULL, &outPath );
                if( result == NFD_OKAY )
                {
                    fmt::print( "Folder path: \"{}\"\n", outPath );
                    free( outPath );
                }
                else if( result == NFD_CANCEL )
                {
                    fmt::print( "User pressed cancel.\n" );
                }
                else
                {
                    fmt::print( "Error: {}\n", NFD_GetError() );
                }
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Take Screenshot" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Edit" ) )
        {
            if( ImGui::MenuItem( "Cut system", "ctrl+x" ) )
            {
                this->cut_image();
            }
            if( ImGui::MenuItem( "Copy system", "ctrl+c" ) )
            {
                Chain_Image_to_Clipboard( this->state.get() );
            }
            if( ImGui::MenuItem( "Paste system", "ctrl+v" ) )
            {
                this->paste_image();
            }
            if( ImGui::MenuItem( "Insert left", "ctrl+leftarrow" ) )
            {
            }
            if( ImGui::MenuItem( "Insert right", "ctrl+rightarrow" ) )
            {
            }
            if( ImGui::MenuItem( "Delete system", "del" ) )
            {
                this->delete_image();
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Controls" ) )
        {
            if( ImGui::MenuItem( "Start/stop calculation", "space" ) )
            {
            }
            if( ImGui::MenuItem( "Randomize spins", "ctrl+r" ) )
            {
            }
            if( ImGui::MenuItem( "Cycle method", "ctrl+m" ) )
            {
            }
            if( ImGui::MenuItem( "Cycle solver", "ctrl+s" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Toggle dragging mode", "F5" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle defect mode", "F6" ) )
            {
            }
            if( ImGui::MenuItem( "Toggle pinning mode", "F7" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "View" ) )
        {
            ImGui::MenuItem( "Show info-widgets", "i", &show_overlays );
            ImGui::MenuItem( "Show settings" );
            ImGui::MenuItem( "Show plots", "", &show_plots );
            ImGui::MenuItem( "Show geometry", "", &show_geometry_settings );
            ImGui::MenuItem( "Show debug" );
            ImGui::Separator();
            if( ImGui::MenuItem( "Regular mode" ) )
            {
            }
            if( ImGui::MenuItem( "Isosurface mode" ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode X" ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode Y" ) )
            {
            }
            if( ImGui::MenuItem( "Slab mode Z" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Toggle camera projection", "c" ) )
            {
            }
            ImGui::Separator();
            if( ImGui::MenuItem( "Toggle visualisation", "ctrl+f" ) )
            {
            }
            if( ImGui::MenuItem( "Fullscreen", "ctrl+shift+f" ) )
            {
            }
            ImGui::EndMenu();
        }
        if( ImGui::BeginMenu( "Help" ) )
        {
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

        width = 2.5f * font_size_px;
        ImGui::SameLine( right_edge - width, 0 );
        if( dark_mode )
        {
            if( ImGui::Button( ICON_FA_SUN, ImVec2( width, bar_height ) ) )
            {
                ImGui::StyleColorsLight();
                dark_mode                         = false;
                rendering_layer.background_colour = glm::vec4{ 0.9f, 0.9f, 0.9f, 1.f };
                rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                    rendering_layer.background_colour );
            }
        }
        else
        {
            if( ImGui::Button( ICON_FA_MOON, ImVec2( width, bar_height ) ) )
            {
                styles::apply_charcoal();
                dark_mode                         = true;
                rendering_layer.background_colour = glm::vec4{ 0.4f, 0.4f, 0.4f, 1.f };
                rendering_layer.view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                    rendering_layer.background_colour );
            }
        }
        right_edge -= ( width + style.FramePadding.x );

        // TODO: deactivate method selection if a calculation is running

        for( int n = modes.size(); n > 0; n-- )
        {
            auto mode         = GUI_Mode( n );
            std::string label = modes[mode].first;
            ImVec2 text_size  = ImGui::CalcTextSize( label.c_str(), NULL, true );
            width             = text_size.x + 2 * style.FramePadding.x;
            height            = text_size.y + 2 * style.FramePadding.y;

            ImGui::SameLine( right_edge - width, 0 );
            if( ImGui::Selectable( label.c_str(), selected_mode == mode, 0, ImVec2( width, height ) ) )
                selected_mode = mode;
            right_edge -= ( width + 2 * style.FramePadding.x );

            if( ImGui::IsItemHovered() )
            {
                // 1.5s delay before showing tooltip
                mode_button_hovered_duration[n - 1] += io.DeltaTime;
                if( mode_button_hovered_duration[n - 1] > 1.5f )
                {
                    ImGui::BeginTooltip();
                    ImGui::Text( modes[mode].second.c_str() );
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
        bool calculation_running
            = Simulation_Running_On_Chain( state.get() ) || Simulation_Running_On_Image( state.get() );
        if( calculation_running )
        {
            if( ImGui::Button( ICON_FA_STOP, ImVec2( width, button_height ) ) )
            {
                this->start_stop();
            }
        }
        else
        {
            if( ImGui::Button( ICON_FA_PLAY, ImVec2( width, button_height ) ) )
            {
                this->start_stop();
            }
        }

        if( ImGui::Button( ICON_FA_ARROW_LEFT, ImVec2( width, button_height ) ) )
        {
        }

        image_number = ( ImU32 )( System_Get_Index( state.get() ) + 1 );
        chain_length = ( ImU32 )( Chain_Get_NOI( state.get() ) );
        ImGui::SetNextItemWidth( 40 );
        if( ImGui::InputScalar(
                "##imagenumber", ImGuiDataType_U32, &image_number, NULL, NULL, "%u",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
            Chain_Jump_To_Image( state.get(), image_number - 1 );
        ImGui::Text( "/" );
        ImGui::SetNextItemWidth( 40 );
        if( ImGui::InputScalar(
                "##chainlength", ImGuiDataType_U32, &chain_length, NULL, NULL, "%u",
                ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            int length = Chain_Get_NOI( state.get() );
            Chain_Set_Length( state.get(), chain_length );
            int new_length = Chain_Get_NOI( state.get() );
            if( new_length < length )
            {
                for( int i = length - 1; i >= new_length; --i )
                    if( threads_image[i].joinable() )
                        threads_image[i].join();
                threads_image.pop_back();
            }
        }

        if( ImGui::Button( ICON_FA_ARROW_RIGHT, ImVec2( width, button_height ) ) )
        {
        }

        ImGui::PopStyleVar(); // ImGuiStyleVar_SelectableTextAlign

        ImGui::EndMainMenuBar();
    }
    ImGui::PopFont();
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

MainWindow::MainWindow( std::shared_ptr<State> state ) : rendering_layer( state )
{
    global_window_handle = this;

    this->state         = state;
    this->threads_image = std::vector<std::thread>( Chain_Get_NOI( this->state.get() ) );

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

    rendering_layer.initialize_gl();
    this->reset_camera();

    // Setup style
    styles::apply_charcoal();

    // Load Fonts
    font_cousine_14 = fonts::cousine( 14 );
    font_karla_14   = fonts::karla( 14 );
    font_karla_16   = fonts::karla( 16 );

    glfwSetScrollCallback( glfw_window, ImGui_ImplGlfw_ScrollCallback );
    glfwSetMouseButtonCallback( glfw_window, ImGui_ImplGlfw_MouseButtonCallback );
    glfwSetKeyCallback( glfw_window, ImGui_ImplGlfw_KeyCallback );
    glfwSetCharCallback( glfw_window, ImGui_ImplGlfw_CharCallback );

    glfwSetFramebufferSizeCallback( glfw_window, framebufferSizeCallback );

#ifdef __EMSCRIPTEN__
    resizeCanvas();
#endif
}

MainWindow::~MainWindow()
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

void MainWindow::resize( int width, int height )
{
    rendering_layer.needs_redraw();
    this->draw();
}