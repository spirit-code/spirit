#ifndef __EMSCRIPTEN__

#include "utility/Handle_Signal.hpp"
#include <lyra/lyra.hpp>

#ifdef SPIRIT_UI_CXX_USE_QT
#include "MainWindow.hpp"
#elif SPIRIT_UI_USE_IMGUI
#include "main_window.hpp"
#endif

#include <Spirit/Chain.h>
#include <Spirit/Configurations.h>
#include <Spirit/IO.h>
#include <Spirit/Log.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/Transitions.h>
#include <Spirit/Version.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <memory>
#include <string>

// Initialise global state pointer
std::shared_ptr<State> state;

// Main
int main( int argc, char ** argv )
{
    // Register interrupt signal
    signal( SIGINT, Utility::Handle_Signal::Handle_SigInt );

    // Default options
    bool show_help    = false;
    bool show_version = false;
    bool quiet        = false;

    std::string cfgfile   = "input/input.cfg";
    std::string imagefile = "";
    std::string chainfile = "";

    // Command line arguments
    auto cli
        = lyra::cli_parser()
          | lyra::opt( cfgfile, "configuration file" )["-f"]["--cfg"]( "The configuration file to use." )
          | lyra::opt( imagefile, "initial image file" )["-i"]["--image"]( "The initial spin configuration to use." )
          | lyra::opt( chainfile, "initial chain file" )["-c"]["--chain"](
              "The initial chain configuration to use. (Overwrites initial spin configuration)" )
          | lyra::opt( quiet )["-q"]["--quiet"]( "If spirit should run in quiet mode." )
          | lyra::opt( show_version )["--version"]( "Show version information." ) | lyra::help( show_help );

    auto result = cli.parse( { argc, argv } );

    if( !result )
    {
        std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
        std::exit( 1 );
    }

    // Show the help when asked for
    if( show_help )
    {
        std::cout << cli << "\n";
        exit( 0 );
    }

    // Show the version when asked for
    if( show_version )
    {
        std::cout << "--------------------------------------\n";
        std::cout << "Spirit Version: " << Spirit_Version_Full() << "\n";
        std::cout << Spirit_Compiler_Full() << "\n";
        std::cout << "--------------------------------------\n";
        std::cout << "scalar_type = " << Spirit_Scalar_Type() << "\n";
        std::cout << "Parallelisation:\n";
        std::cout << "   - OpenMP  = " << Spirit_OpenMP() << "\n";
        std::cout << "   - Cuda    = " << Spirit_Cuda() << "\n";
        std::cout << "   - Threads = " << Spirit_Threads() << "\n";
        std::cout << "Other:\n";
        std::cout << "   - Defects = " << Spirit_Defects() << "\n";
        std::cout << "   - Pinning = " << Spirit_Pinning() << "\n";
        std::cout << "   - FFTW    = " << Spirit_FFTW() << "\n";
        exit( 0 );
    }

    // Initialise state
    state = std::shared_ptr<State>( State_Setup( cfgfile.c_str(), quiet ), State_Delete );

    // Standard initial spin configuration
    Configuration_PlusZ( state.get() );

    // Read image from file
    if( !imagefile.empty() )
        IO_Image_Read( state.get(), imagefile.c_str(), 0 );

    // Read chain from file
    if( !chainfile.empty() )
        IO_Chain_Read( state.get(), chainfile.c_str() );

#ifdef _OPENMP
    Log_Send(
        state.get(), Log_Level_Info, Log_Sender_UI,
        ( "Using OpenMP with " + std::to_string( Spirit_OpenMP_Get_Num_Threads() ) + "/"
          + std::to_string( omp_get_max_threads() ) + " threads" )
            .c_str() );
#endif

#if defined( SPIRIT_UI_CXX_USE_QT )
    //------------------------ User Interface ---------------------------------------
    // Initialise Application and MainWindow
    QApplication app( argc, argv );
    // app.setOrganizationName("--");
    // app.setApplicationName("Spirit - Atomistic Spin Code - OpenGL with Qt");

    // Format for all GL surfaces
    QSurfaceFormat format;
    format.setSamples( 16 );
    format.setVersion( 3, 3 );
    // format.setVersion(4, 2);
    // glFormat.setVersion( 3, 3 );
    // glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
    // glFormat.setSampleBuffers( true );
    format.setProfile( QSurfaceFormat::CoreProfile );
    format.setDepthBufferSize( 24 );
    format.setStencilBufferSize( 8 );
    QSurfaceFormat::setDefaultFormat( format );
    Log_Send(
        state.get(), Log_Level_Info, Log_Sender_UI,
        ( "QSurfaceFormat version: " + std::to_string( format.majorVersion() ) + "."
          + std::to_string( format.minorVersion() ) )
            .c_str() );

    MainWindow window( state );
    window.setWindowTitle( app.applicationName() );
    window.show();
    // Open the application
    int exec = app.exec();
    // If application is closed normally
    if( exec != 0 )
        throw exec;
    // Finish
    state.reset();
    return exec;
    //-------------------------------------------------------------------------------
#elif defined( SPIRIT_UI_USE_IMGUI )
    ui::MainWindow window( state );

    // Open the Application
    int exec = window.run();
    // If Application is closed normally
    if( exec != 0 )
        throw exec;
    // Finish
    state.reset();
    return exec;
#else
    //----------------------- LLG iterations ----------------------------------------
    Simulation_LLG_Start( state.get(), Solver_SIB );
    //-------------------------------------------------------------------------------
#endif

    state.reset();
    return 0;
}

#endif