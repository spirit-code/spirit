#include <fonts.hpp>
#include <main_window.hpp>
#include <styles.hpp>
#include <widgets.hpp>

#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/BoundingBoxRenderer.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/CoordinateSystemRenderer.hxx>
#include <VFRendering/IsosurfaceRenderer.hxx>
#include <VFRendering/SphereRenderer.hxx>
#include <VFRendering/View.hxx>

#include <imgui_impl/glfw.h>
#include <imgui_impl/opengl3.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <fmt/format.h>

#include <stdio.h>
#include <cmath>

static GLFWwindow * g_window;
static ImVec4 background_colour = ImVec4( 0.4f, 0.4f, 0.4f, 1.f );
static bool show_demo_window    = true;
static bool show_another_window = false;
static int selected_mode        = 1;

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
static VFRendering::VectorField vfr_vectorfield = VFRendering::VectorField( {}, {} );
static std::shared_ptr<VFRendering::ArrowRenderer> vfr_arrow_renderer_ptr;

/////////////////////////////////////////////////////////////////////

#ifdef __EMSCRIPTEN__
EM_JS( int, canvas_get_width, (), { return Module.canvas.width; } );
EM_JS( int, canvas_get_height, (), { return Module.canvas.height; } );
EM_JS( void, resizeCanvas, (), { js_resizeCanvas(); } );
#endif

static void glfw_error_callback( int error, const char * description )
{
    std::cerr << fmt::format( "Glfw Error {}: {}\n", error, description );
}

void mouseWheelCallback( GLFWwindow * window, double x_offset, double y_offset )
{
    (void)window;
    (void)x_offset;
    float scale = 10;
    // TODO:
    // if( shift_pressed )
    //     scale = 1;
    vfr_view.mouseScroll( -scale * y_offset );
    // needs_redraw = true;
}

void mousePositionCallback( GLFWwindow * window, double x_position, double y_position )
{
    static glm::vec2 previous_mouse_position( 0, 0 );

    if( ImGui::GetIO().WantCaptureMouse )
        return;

    glm::vec2 current_mouse_position( x_position, y_position );
    if( glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_LEFT ) == GLFW_PRESS )
    {
        auto movement_mode = VFRendering::CameraMovementModes::ROTATE_BOUNDED;
        if( glfwGetKey( window, GLFW_KEY_LEFT_CONTROL ) == GLFW_PRESS )
        {
            movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
        }
        vfr_view.mouseMove( previous_mouse_position, current_mouse_position, movement_mode );
        // needs_redraw = true;
    }
    previous_mouse_position = current_mouse_position;
}

void framebufferSizeCallback( GLFWwindow * window, int width, int height )
{
    (void)window;
    vfr_view.setFramebufferSize( width, height );
    // needs_redraw = true;
}

void keyCallback( GLFWwindow * window, int key, int scancode, int action, int mods )
{
    (void)window;
    (void)scancode;
    if( mods != 0 )
    {
        return;
    }
    if( action != GLFW_PRESS && action != GLFW_REPEAT )
    {
        return;
    }
    switch( key )
    {
        case GLFW_KEY_R:
        {
            VFRendering::Options options;
            options.set<VFRendering::View::Option::CAMERA_POSITION>( { 0, 0, 30 } );
            options.set<VFRendering::View::Option::CENTER_POSITION>( { 0, 0, 0 } );
            options.set<VFRendering::View::Option::UP_VECTOR>( { 0, 1, 0 } );
            vfr_view.updateOptions( options );
        }
        // needs_redraw = true;
        break;
    }
}

void windowRefreshCallback( GLFWwindow * window )
{
    (void)window;
    // needs_redraw = true;
}

/////////////////////////////////////////////////////////////////////

void main_window::intitialize_gl()
{
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CONE_RADIUS>( 0.125f );
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CONE_HEIGHT>( 0.3f );
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_RADIUS>( 0.0625f );
    vfr_view.setOption<VFRendering::ArrowRenderer::Option::CYLINDER_HEIGHT>( 0.35f );
    vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
        { background_colour.x, background_colour.y, background_colour.z } );

    vfr_arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>( vfr_view, vfr_vectorfield );

    int n_cells[3]                      = { 10, 10, 10 };
    std::vector<glm::vec3> positions    = std::vector<glm::vec3>( n_cells[0] * n_cells[1] * n_cells[2] );
    std::vector<glm::vec3> orientations = std::vector<glm::vec3>( n_cells[0] * n_cells[1] * n_cells[2] );
    for( int cell_c = 0; cell_c < n_cells[2]; cell_c++ )
    {
        for( int cell_b = 0; cell_b < n_cells[1]; cell_b++ )
        {
            for( int cell_a = 0; cell_a < n_cells[0]; cell_a++ )
            {
                int idx           = cell_a + n_cells[1] * cell_b + n_cells[1] * n_cells[2] * cell_c;
                positions[idx]    = glm::vec3( cell_a, cell_b, cell_c );
                orientations[idx] = glm::vec3( 0, 0, 1 );
            }
        }
    }
    // vfr_geometry = VFRendering::Geometry( positions, {}, {}, false );
    vfr_geometry = VFRendering::Geometry::cartesianGeometry(
        { n_cells[0], n_cells[1], n_cells[2] }, { 0, 0, 0 }, { 10, 10, 10 } );

    // VFRendering::Geometry geometry = VFRendering::Geometry::cartesianGeometry({21, 21, 21}, {-20, -20, -20}, {20, 20,
    // 20}); VFRendering::VectorField vf = VFRendering::VectorField(geometry, directions);

    vfr_vectorfield.updateGeometry( vfr_geometry );
    vfr_vectorfield.updateVectors( orientations );

    VFRendering::Options options;
    auto center = ( vfr_geometry.min() + vfr_geometry.max() ) * 0.5f;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>( center );
    // options.set<VFRendering::View::Option::SYSTEM_CENTER>( { 0, 0, 0 } );
    options.set<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(
        VFRendering::Utilities::getColormapImplementation( VFRendering::Utilities::Colormap::HSV ) );
    options.set<VFRendering::View::Option::CAMERA_POSITION>( { 0, 0, 30 } );
    options.set<VFRendering::View::Option::CENTER_POSITION>( options.get<VFRendering::View::Option::SYSTEM_CENTER>() );
    options.set<VFRendering::View::Option::UP_VECTOR>( { 0, 0, 1 } );
    vfr_view.updateOptions( options );

    std::cerr << "min       " << vfr_geometry.min().x << " " << vfr_geometry.min().y << " " << vfr_geometry.min().z
              << "\n";
    std::cerr << "max       " << vfr_geometry.max().x << " " << vfr_geometry.max().y << " " << vfr_geometry.max().z
              << "\n";
    std::cerr << "center at " << center.x << " " << center.y << " " << center.z << "\n";
    auto cam = options.get<VFRendering::View::Option::CENTER_POSITION>();
    std::cerr << "camera at " << cam.x << " " << cam.y << " " << cam.z << "\n";

    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers = { vfr_arrow_renderer_ptr };
    vfr_view.renderers(
        { { std::make_shared<VFRendering::CombinedRenderer>( vfr_view, renderers ), { { 0, 0, 1, 1 } } } } );
}

void main_window::draw_gl( int display_w, int display_h )
{
    glViewport( 0, 0, display_w, display_h );
    vfr_view.setFramebufferSize( float( display_w ), float( display_h ) );
    vfr_view.draw();
}

int main_window::run()
{
    while( !glfwWindowShouldClose( g_window ) )
    {
#ifdef __EMSCRIPTEN__
        int width  = canvas_get_width();
        int height = canvas_get_height();

        glfwSetWindowSize( g_window, width, height );
#endif

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::PushFont( font_14 );

        widgets::show_menu_bar( g_window, font_16, dark_mode, background_colour, selected_mode, vfr_view );
        bool p_open = true;
        widgets::show_overlay( &p_open );

        if( show_another_window )
        {
            ImGui::Begin( "Another Window", &show_another_window );
            ImGui::Text( "Hello from another window!" );
            if( ImGui::Button( "Close Me" ) )
                show_another_window = false;
            ImGui::End();
        }

        if( show_demo_window )
        {
            ImGui::SetNextWindowPos( ImVec2( 650, 20 ), ImGuiCond_FirstUseEver );
            ImGui::ShowDemoWindow( &show_demo_window );
        }

        {
            static float f     = 0.0f;
            static int counter = 0;
            ImGui::Text( "Hello, world!" );
            ImGui::SliderFloat( "float", &f, 0.0f, 1.0f );
            if( ImGui::ColorEdit3( "clear color", (float *)&background_colour ) )
            {
                vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
                    { background_colour.x, background_colour.y, background_colour.z } );
            }

            ImGui::Text( "Windows" );
            ImGui::Checkbox( "Demo Window", &show_demo_window );
            ImGui::Checkbox( "Another Window", &show_another_window );

            if( ImGui::Button( "Button" ) )
                counter++;
            ImGui::SameLine();
            ImGui::Text( "counter = %d", counter );

            ImGui::Text(
                "Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                ImGui::GetIO().Framerate );

            static int antialiasing = 0;
            ImGui::Text( fmt::format( "{}x antialiasing", antialiasing ).c_str() );
            ImGui::SameLine();
            ImGui::SliderInt( "antialiasing", &antialiasing, 0, 16 );
            // glfwWindowHint( GLFW_SAMPLES, antialiasing ); // 16x antialiasing

            if( ImGui::Button( "apply antialiasing" ) )
                glfwWindowHint( GLFW_SAMPLES, antialiasing ); // 16x antialiasing
        }

        ImGui::PopFont();

        ImGui::Render();

        int display_w, display_h;
        glfwMakeContextCurrent( g_window );
        glfwGetFramebufferSize( g_window, &display_w, &display_h );

        draw_gl( display_w, display_h );

        ImGui_ImplOpenGL3_RenderDrawData( ImGui::GetDrawData() );

        glfwSwapBuffers( g_window );
    }
    return 0;
}

main_window::main_window( std::shared_ptr<State> state )
{
    this->state = state;

    glfwSetErrorCallback( glfw_error_callback );

    if( !glfwInit() )
    {
        std::cerr << fmt::format( "Failed to initialize GLFW\n" );
        // return 1;
        throw;
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
    g_window         = glfwCreateWindow( canvasWidth, canvasHeight, "Spirit - Magnetism Simulation Tool", NULL, NULL );
    glfwMakeContextCurrent( g_window );
    glfwSwapInterval( 1 ); // Enable vsync

    if( g_window == NULL )
    {
        std::cerr << fmt::format( "Failed to open GLFW window.\n" );
        glfwTerminate();
        // return -1;
        throw;
    }

    gladLoadGL( (GLADloadfunc)glfwGetProcAddress ); // Initialize GLAD

    // Setup Dear ImGui binding
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io   = ImGui::GetIO();
    io.IniFilename = "imgui_state.ini";

    ImGui_ImplGlfw_InitForOpenGL( g_window, false );
    ImGui_ImplOpenGL3_Init();

    intitialize_gl();

    // Setup style
    styles::apply_charcoal();

    // Load Fonts
    font_14 = fonts::font_combined( 14 );
    font_16 = fonts::font_combined( 16 );
    font_18 = fonts::font_combined( 18 );

    glfwSetScrollCallback( g_window, mouseWheelCallback );
    glfwSetCursorPosCallback( g_window, mousePositionCallback );
    glfwSetFramebufferSizeCallback( g_window, framebufferSizeCallback );
    glfwSetKeyCallback( g_window, keyCallback );

#ifdef __EMSCRIPTEN__
    resizeCanvas();
#endif
}