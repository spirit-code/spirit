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

#include <fmt/format.h>

#include <exception>

static GLFWwindow * glfw_window;
static ImVec4 background_colour = ImVec4( 0.4f, 0.4f, 0.4f, 0.f );
static bool show_demo_window    = false;
static bool show_keybindings    = false;
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
static VFRendering::VectorField vfr_vectorfield = VFRendering::VectorField( {}, {} );
// static std::shared_ptr<VFRendering::ArrowRenderer> vfr_arrow_renderer_ptr;
static std::vector<std::shared_ptr<VFRendering::RendererBase>> vfr_renderers( 0 );

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

void mouseWheelCallback( GLFWwindow * window, double x_offset, double y_offset )
{
    if( ImGui::GetIO().WantCaptureMouse )
        return;

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
        vfr_view.mouseMove( previous_mouse_position, current_mouse_position, movement_mode );
        // needs_redraw = true;
    }
    else if( glfwGetMouseButton( window, GLFW_MOUSE_BUTTON_RIGHT ) == GLFW_PRESS )
    {
        auto movement_mode = VFRendering::CameraMovementModes::TRANSLATE;
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

#ifdef __EMSCRIPTEN__
    resizeCanvas();
#endif
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

    int n_cells[3]                      = { 6, 6, 6 };
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
        { n_cells[0], n_cells[1], n_cells[2] }, { 0, 0, 0 }, { n_cells[0], n_cells[1], n_cells[2] } );

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
    options.set<VFRendering::View::Option::CAMERA_POSITION>( { -20, -20, 20 } );
    options.set<VFRendering::View::Option::CENTER_POSITION>( options.get<VFRendering::View::Option::SYSTEM_CENTER>() );
    options.set<VFRendering::View::Option::UP_VECTOR>( { 0, 0, 1 } );
    options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( 45 );
    options.set<VFRendering::View::Option::VERTICAL_FIELD_OF_VIEW>( 45 );
    vfr_view.updateOptions( options );

    fmt::print( "min       {} {} {}\n", vfr_geometry.min().x, vfr_geometry.min().y, vfr_geometry.min().z );
    fmt::print( "max       {} {} {}\n", vfr_geometry.max().x, vfr_geometry.max().y, vfr_geometry.max().z );
    auto sys_center = options.get<VFRendering::View::Option::SYSTEM_CENTER>();
    fmt::print( "system center at {} {} {}\n", sys_center.x, sys_center.y, sys_center.z );
    auto cam_center = options.get<VFRendering::View::Option::CENTER_POSITION>();
    fmt::print( "camera center at {} {} {}\n", cam_center.x, cam_center.y, cam_center.z );
    auto cam = options.get<VFRendering::View::Option::CAMERA_POSITION>();
    fmt::print( "camera position at {} {} {}\n", cam.x, cam.y, cam.z );

    auto vfr_arrow_renderer_ptr = std::make_shared<VFRendering::ArrowRenderer>( vfr_view, vfr_vectorfield );
    vfr_renderers.push_back( vfr_arrow_renderer_ptr );

    vfr_view.renderers(
        { { std::make_shared<VFRendering::CombinedRenderer>( vfr_view, vfr_renderers ), { { 0, 0, 1, 1 } } } } );
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

    draw_imgui( display_w, display_h );

    glfwSwapBuffers( glfw_window );

#ifdef __EMSCRIPTEN__
    emscripten_webgl_make_context_current( context_vfr );
#endif

    draw_vfr( display_w, display_h );
}

void main_window::draw_vfr( int display_w, int display_h )
{
    vfr_view.setFramebufferSize( float( display_w ), float( display_h ) );
    vfr_view.draw();
}

void main_window::draw_imgui( int display_w, int display_h )
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::PushFont( font_14 );

    widgets::show_menu_bar( glfw_window, font_16, dark_mode, background_colour, selected_mode, vfr_view );
    bool p_open = true;
    widgets::show_overlay( &p_open );

    if( show_keybindings )
    {
        ImGui::Begin( "Keybindings", &show_keybindings );
        ImGui::Text( "F1: this window" );
        ImGui::Text( "1-5: Choose mode" );
        ImGui::Text( "c: switch camera projection (perspective/parallel)" );
        ImGui::Text( "wasd: ..." );
        ImGui::Text( "arrows: ..." );
        if( ImGui::Button( "Close" ) )
            show_keybindings = false;
        ImGui::End();
    }

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
    if( ImGui::ColorEdit3( "clear color", (float *)&background_colour ) )
    {
        vfr_view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(
            { background_colour.x, background_colour.y, background_colour.z } );
    }

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

    this->state = state;

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

    glfwSetScrollCallback( glfw_window, mouseWheelCallback );
    glfwSetCursorPosCallback( glfw_window, mousePositionCallback );
    glfwSetFramebufferSizeCallback( glfw_window, framebufferSizeCallback );
    glfwSetKeyCallback( glfw_window, keyCallback );

#ifdef __EMSCRIPTEN__
    resizeCanvas();
#endif
}