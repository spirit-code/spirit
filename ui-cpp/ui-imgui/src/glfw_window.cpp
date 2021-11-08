#include <glfw_window.hpp>

#include <fmt/format.h>

#include <exception>

static void glfw_error_callback( int error, const char * description )
{
    fmt::print( "Glfw Error {}: {}\n", error, description );
}

namespace ui
{

GlfwWindow::GlfwWindow( const std::string & title )
{
    glfwSetErrorCallback( glfw_error_callback );

    // Note: for a macOS .app bundle, `glfwInit` changes the working
    // directory to the bundle's Contents/Resources directory
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
    // glfwWindowHint( GLFW_AUTO_ICONFIY, false );
    // glfwWindowHint( GLFW_DECORATED, false );
    // glfwWindowHint( GLFW_RESIZABLE, true );
#if __APPLE__
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, true );
#endif

    // Open a window and create its OpenGL context
    int canvas_width  = 1280;
    int canvas_height = 720;
    glfw_window       = glfwCreateWindow( canvas_width, canvas_height, title.c_str(), nullptr, nullptr );
    glfwMakeContextCurrent( glfw_window );
#ifndef __EMSCRIPTEN__
    glfwSwapInterval( 1 ); // Enable vsync
#endif

    if( glfw_window == nullptr )
    {
        fmt::print( "Failed to open GLFW window.\n" );
        glfwTerminate();
        throw std::runtime_error( "Failed to open GLFW window." );
    }
}

GlfwWindow::~GlfwWindow()
{
    glfwDestroyWindow( glfw_window );
    glfwTerminate();
}

} // namespace ui