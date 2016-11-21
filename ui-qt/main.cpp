#include "MainWindow.hpp"

#include "Interface_State.h"
#include "Interface_Chain.h"
#include "Interface_Configurations.h"
#include "Interface_Transitions.h"
#include "Interface_Log.h"

// Main
int main(int argc, char ** argv)
{
	//--- Register SigInt
	// signal(SIGINT, Signal::Handle_SigInt);
	
	//---------------------- file names ---------------------------------------------
	//--- Config Files
	// const char * cfgfile = "markus.cfg";
	// const char * cfgfile = "input/markus-paper.cfg";
	// const char * cfgfile = "input/gideon-master-thesis-isotropic.cfg";
	// const char * cfgfile = "input/gideon-master-thesis-anisotropic.cfg";
	const char * cfgfile = "input/example-hopfion-anisotropic.cfg";
	// const char * cfgfile = "input/kagome-spin-ice.cfg";
	// const char * cfgfile = "input/gaussian/example-1.cfg";
	// const char * cfgfile = "input/gaussian/gideon-paper.cfg";
	// const char * cfgfile = "input/daniel-master-thesis-isotropic.cfg";
	//--- Data Files
	// std::string spinsfile = "input/anisotropic/achiral.txt";
	// std::string chainfile = "input/chain.txt";
	//-------------------------------------------------------------------------------
	
	//--- Initialise State
	std::shared_ptr<State> state = std::shared_ptr<State>(State_Setup(cfgfile), State_Delete);

	//---------------------- initialize spin_systems --------------------------------
	// Copy the system a few times
	/*Chain_Image_to_Clipboard(state.get());
	for (int i=1; i<7; ++i)
	{
		Chain_Insert_Image_After(state.get());
	}*/
	//-------------------------------------------------------------------------------
	
	//----------------------- spin_system_chain -------------------------------------
	// Parameters
	float dir[3] = { 0,0,1 };
	float pos[3] = { 0,0,0 };

	// Read Image from file
	//Configuration_from_File(state.get(), spinsfile, 0);
	// Read Chain from file
	//Chain_from_File(state.get(), chainfile);

	// First image is homogeneous with a Skyrmion at pos
	Configuration_Homogeneous(state.get(), dir, 0);
	Configuration_Skyrmion(state.get(), pos, 6.0, 1.0, -90.0, false, false, false, 0);
	// Last image is homogeneous
	Configuration_Homogeneous(state.get(), dir, Chain_Get_NOI(state.get())-1);

	// Create transition of images between first and last
	Transition_Homogeneous(state.get(), 0, Chain_Get_NOI(state.get())-1);

	// Update the Chain's Data'
	Chain_Update_Data(state.get());
	//-------------------------------------------------------------------------------

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
	if (exec == 0)
	{
		Log_Send(state.get(), Log_Level_All, Log_Sender_All, "=====================================================");
		Log_Send(state.get(), Log_Level_All, Log_Sender_All, "================= Spirit Finished ===================");
		Log_Send(state.get(), Log_Level_All, Log_Sender_All, "=====================================================");
		Log_Append(state.get());
	}
	else throw exec;
	// Finish
	return exec;
	//-------------------------------------------------------------------------------


	return 0;
}
