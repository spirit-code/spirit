#include "utility/CommandLineParser.hpp"
#include "utility/Handle_Signal.hpp"

#include "Spirit/State.h"
#include "Spirit/Chain.h"
#include "Spirit/Configurations.h"
#include "Spirit/Transitions.h"
#include "Spirit/Simulation.h"
#include "Spirit/Log.h"

#ifdef _OPENMP
    #include <omp.h>
#endif

#ifdef UI_CXX_USE_QT
    #include "MainWindow.hpp"
#endif

#include <memory>
#include <string>

// Initialise global state pointer
std::shared_ptr<State> state;

// Main
int main(int argc, char ** argv)
{
    //--- Register SigInt
    signal(SIGINT, Utility::Handle_Signal::Handle_SigInt);

    //--- Default config file
    std::string cfgfile = "input/input.cfg";

    //--- Command line arguments
    CommandLineParser cmdline(argc, argv);
    // Quiet run
    bool quiet = cmdline.cmdOptionExists("-quiet");
    // Config file
    const std::string & filename = cmdline.getCmdOption("-f");
    if( !filename.empty() )
        cfgfile = filename;

    //--- Data Files
    // std::string spinsfile = "input/anisotropic/achiral.txt";
    // std::string chainfile = "input/chain.txt";

    //--- Initialise State
    state = std::shared_ptr<State>(State_Setup(cfgfile.c_str(), quiet), State_Delete);

    //--- Initial spin configuration
    Configuration_PlusZ(state.get());
    // // Read Image from file
    // Configuration_from_File(state.get(), spinsfile, 0);

    //--- Chain
    // // Read Chain from file
    // Chain_from_File(state.get(), chainfile);

    // // Set the chain length
    // Chain_Set_Length(state.get(), 12);

    // // First image is plus-z with a Bloch skyrmion at the center
    // Configuration_PlusZ(state.get());
    // Configuration_Skyrmion(state.get(), 6.0, 1.0, -90.0, false, false, false);

    // // Last image is plus-z
    // Chain_Jump_To_Image(state.get(), Chain_Get_NOI(state.get())-1);
    // Configuration_PlusZ(state.get());
    // Chain_Jump_To_Image(state.get(), 0);

    // // Create transition of images between first and last
    // Transition_Homogeneous(state.get(), 0, Chain_Get_NOI(state.get())-1);

    // // Update the Chain's Data'
    // Chain_Update_Data(state.get());
    //-------------------------------------------------------------------------------

    #ifdef _OPENMP
        int nt = omp_get_max_threads() - 1;
        Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("Using OpenMP with n=" + std::to_string(nt) + " threads").c_str());
    #endif

    #ifdef UI_CXX_USE_QT
        //------------------------ User Interface ---------------------------------------
        // Initialise Application and MainWindow
        QApplication app(argc, argv);
        //app.setOrganizationName("--");
        //app.setApplicationName("Spirit - Atomistic Spin Code - OpenGL with Qt");

        // Format for all GL Surfaces
        QSurfaceFormat format;
        format.setSamples(16);
        format.setVersion(3, 3);
        //format.setVersion(4, 2);
        //glFormat.setVersion( 3, 3 );
        //glFormat.setProfile( QGLFormat::CoreProfile ); // Requires >=Qt-4.8.0
        //glFormat.setSampleBuffers( true );
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setDepthBufferSize(24);
        format.setStencilBufferSize(8);
        QSurfaceFormat::setDefaultFormat(format);
        Log_Send(state.get(), Log_Level_Info, Log_Sender_UI, ("QSurfaceFormat version: " + std::to_string(format.majorVersion()) + "." + std::to_string(format.minorVersion())).c_str());

        MainWindow window(state);
        window.setWindowTitle(app.applicationName());
        window.show();
        // Open the Application
        int exec = app.exec();
        // If Application is closed normally
        if (exec != 0) throw exec;
        // Finish
        state.reset();
        return exec;
        //-------------------------------------------------------------------------------
    #else
        //----------------------- LLG Iterations ----------------------------------------
        Simulation_LLG_Start(state.get(), Solver_SIB);
        //-------------------------------------------------------------------------------
    #endif

    state.reset();
    return 0;
}
