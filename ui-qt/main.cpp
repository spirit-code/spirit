#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <map>

#include <QtGui/QGuiApplication>
#include <QApplication>

#include "MainWindow.h"

#include "Version.h"
#include "Geometry.h"
#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Configurations.h"
#include "Configuration_Chain.h"

#include "Solver_LLG.h"
#include "Solver_GNEB.h"

#include "IO.h"
#include "Logging.h"
#include "Threading.h"
#include "Exception.h"
#include "Signal.h"

#include "Interface_State.h"

// Use Core Namespaces
using namespace Data;
using namespace Engine;
using namespace Utility;

// Initialise Global Variables
std::shared_ptr<State> state;
Utility::LoggingHandler Utility::Log = Utility::LoggingHandler(Log_Level::WARNING, Log_Level::DEBUG, ".", "Log_" + Timing::CurrentDateTime() + ".txt");
std::map<std::shared_ptr<Data::Spin_System>, std::thread> Utility::Threading::llg_threads = std::map<std::shared_ptr<Data::Spin_System>, std::thread>();
std::map<std::shared_ptr<Data::Spin_System_Chain>, std::thread> Utility::Threading::gneb_threads = std::map<std::shared_ptr<Data::Spin_System_Chain>, std::thread>();

// Main
int main(int argc, char ** argv)
{
	// Register SigInt
	signal(SIGINT, Signal::Handle_SigInt);
	
	//--- Config Files
	//const char * cfgfile = "input/markus-paper.cfg";
	const char * cfgfile = "input/gideon-master-thesis-isotropic.cfg";
	// const char * cfgfile = "input/daniel-master-thesis-isotropic.cfg";

	//--- Initialise State
	state = std::shared_ptr<State>(setupState(cfgfile));
	
	//---------------------- initialize spin_system_parts ---------------------------
	std::string spinsfile = "input/anisotropic/achiral.txt";
	//std::unique_ptr<Data::Debug_Parameters> debug = IO::Debug_Parameters_from_Config(cfgfile);
	//std::unique_ptr<Data::Geometry> geom = IO::Geometry_from_Config(cfgfile);
	//std::unique_ptr<Data::LLG_Parameters> llg = IO::LLG_Parameters_from_Config(cfgfile);
	//std::unique_ptr<Data::Hamiltonian_Isotropic> ham_iso = IO::Hamiltonian_Isotropic_from_Config(cfgfile, *geom, *debug);	// sowas ist KOKOLORES !!! daf�r gibts shared_ptr...
	//std::unique_ptr<Data::Hamiltonian_Anisotropic> ham_aniso = IO::Hamiltonian_Anisotropic_from_Config(cfgfile);
	//-------------------------------------------------------------------------------
	
	//---------------------- initialize spin_systems --------------------------------
	// Create a system according to Config
	auto s1 = state->active_image;
	// Copy the system a few times
	auto s2 = std::shared_ptr<Spin_System>(new Spin_System(*s1));
	auto s3 = std::shared_ptr<Spin_System>(new Spin_System(*s1));
	auto s4 = std::shared_ptr<Spin_System>(new Spin_System(*s1));
	auto s5 = std::shared_ptr<Spin_System>(new Spin_System(*s1));
	auto s6 = std::shared_ptr<Spin_System>(new Spin_System(*s1));
	auto s7 = std::shared_ptr<Spin_System>(new Spin_System(*s1));
	//-------------------------------------------------------------------------------


	//---------------------- set images' configurations -----------------------------
	// Parameters
	double dir[3] = { 0,0,1 };
	std::vector<double> pos = { 14.5, 14.5, 0 };
	// Read Image from file
	//Utility::IO::Read_Spin_Configuration(s1, spinsfile);
	// First image is homogeneous with a Skyrmion at pos
	Configurations::Homogeneous(*s1, dir);
	Configurations::Skyrmion(*s1, pos, 6.0, 1.0, -90.0, false, false, false, false);
	// Las image is homogeneous
	Configurations::Homogeneous(*s7, dir);
	//-------------------------------------------------------------------------------

	
	//----------------------- spin_system_chain -------------------------------------
	// Get parameters
	auto params_gneb = std::shared_ptr<Parameters_GNEB>(IO::GNEB_Parameters_from_Config(cfgfile));
	// Create the chain
	auto sv = std::vector<std::shared_ptr<Data::Spin_System>>();
	sv.push_back(s1);
	sv.push_back(s2);
	sv.push_back(s3);
	sv.push_back(s4);
	sv.push_back(s5);
	sv.push_back(s6);
	sv.push_back(s7);
	// TODO: use interface function
	state->active_chain = std::shared_ptr<Data::Spin_System_Chain>(new Data::Spin_System_Chain(sv, params_gneb, false));
	state->noi = sv.size();
	// Create transition of images
	Utility::Configuration_Chain::Homogeneous_Rotation(state->active_chain, s1->spins, s7->spins);
	//-------------------------------------------------------------------------------
	
	//------------------------ User Interface ---------------------------------------
	// Initialise Application and MainWindow
	QApplication app(argc, argv);
	//app.setOrganizationName("Forschungszentrum Juelich");
	//app.setApplicationName("MonoSpin - Juelich Spin Code - OpenGL with Qt");

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
	qDebug() << "surface format:" << format.majorVersion() << "." << format.minorVersion();

	MainWindow window(state);
	window.setWindowTitle(app.applicationName());
	window.show();
	// Open the Application
	int exec = app.exec();
	// If Application is closed normally
	if (exec == 0)
	{
		Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
		Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= MonoSpin Finished =================");
		Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
		Log.Append_to_File();
	}
	else throw exec;
	// Finish
	return exec;
	//-------------------------------------------------------------------------------


	return 0;
}