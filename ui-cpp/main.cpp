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

// Initialise global state pointer
std::shared_ptr<State> state;

// Main
int main(int argc, char ** argv)
{
	// Put arguments into the parser
	CommandLineParser cmdline(argc, argv);
    
	//--- Register SigInt
	signal(SIGINT, Utility::Handle_Signal::Handle_SigInt);
	
	//--- Option for output-less State
	bool quiet = cmdline.cmdOptionExists("-quiet");

	//---------------------- file names ---------------------------------------------
	const char * cfgfile;
	//--- Default config file
	cfgfile = "input/input.cfg";
	// cfgfile = "input/anisotropic/markus.cfg";
	// cfgfile = "input/anisotropic/markus-paper.cfg";
	// cfgfile = "input/anisotropic/kagome-spin-ice.cfg";
	// cfgfile = "input/anisotropic/gideon-master-thesis-anisotropic.cfg";
	// cfgfile = "input/isotropic/gideon-master-thesis-isotropic.cfg";
	// cfgfile = "input/isotropic/daniel-master-thesis-isotropic.cfg";
	// cfgfile = "input/gaussian/example-1.cfg";
	// cfgfile = "input/gaussian/gideon-paper.cfg";
	//--- Command line passed config file
    const std::string &filename = cmdline.getCmdOption("-f");
    if (!filename.empty()) cfgfile = filename.c_str();
	//--- Data Files
	// std::string spinsfile = "input/anisotropic/achiral.txt";
	// std::string chainfile = "input/chain.txt";
	//-------------------------------------------------------------------------------
	
	//--- Initialise State
	state = std::shared_ptr<State>(State_Setup(cfgfile, quiet), State_Delete);

	//---------------------- initialize spin_systems --------------------------------
	// Copy the system a few times
	/*Chain_Image_to_Clipboard(state.get());
	for (int i=1; i<7; ++i)
	{
		Chain_Insert_Image_After(state.get());
	}*/
	//-------------------------------------------------------------------------------
	
	//----------------------- spin_system_chain -------------------------------------
	// Read Image from file
	//Configuration_from_File(state.get(), spinsfile, 0);
	// Read Chain from file
	//Chain_from_File(state.get(), chainfile);

	// First image is homogeneous with a Skyrmion at pos
	Configuration_PlusZ(state.get());
	Configuration_Skyrmion(state.get(), 6.0, 1.0, -90.0, false, false, false);
	// Last image is homogeneous
	Chain_Jump_To_Image(state.get(), Chain_Get_NOI(state.get())-1);
	Configuration_PlusZ(state.get());
	Chain_Jump_To_Image(state.get(), 0);

	// Create transition of images between first and last
	Transition_Homogeneous(state.get(), 0, Chain_Get_NOI(state.get())-1);

	// Update the Chain's Data'
	Chain_Update_Data(state.get());
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
		return exec;
		//-------------------------------------------------------------------------------
	#else
		//----------------------- LLG Iterations ----------------------------------------
		Simulation_PlayPause(state.get(), "LLG", "SIB");
		//-------------------------------------------------------------------------------
	#endif


	return 0;
}
