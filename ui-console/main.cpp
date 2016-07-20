#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <map>

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

// Use Core Namespaces
using namespace Data;
using namespace Engine;
using namespace Utility;

// Initialise Global Variables
std::shared_ptr<Data::Spin_System_Chain> c;
Utility::LoggingHandler Utility::Log = Utility::LoggingHandler(Log_Level::WARNING, Log_Level::DEBUG, ".", "Log_" + Timing::CurrentDateTime() + ".txt");
std::map<std::shared_ptr<Data::Spin_System>, std::thread> Utility::Threading::llg_threads = std::map<std::shared_ptr<Data::Spin_System>, std::thread>();
std::map<std::shared_ptr<Data::Spin_System_Chain>, std::thread> Utility::Threading::gneb_threads = std::map<std::shared_ptr<Data::Spin_System_Chain>, std::thread>();

// Main
int main(int argc, char ** argv)
{
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "====================================================");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= MonoSpin Started =================");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= Version:  " + std::string(VERSION));
	Log.Send(Log_Level::INFO, Log_Sender::ALL, "================= Revision: " + std::string(VERSION_REVISION));
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "====================================================");

	try {
		//--- Config Files
		//std::string cfgfile = "input/markus-paper.cfg";
		std::string cfgfile = "input/gideon-master-thesis-isotropic.cfg";
		// std::string cfgfile = "input/gideon-master-thesis-anisotropic.cfg";
		//--- Read Log Levels
		IO::Log_Levels_from_Config(cfgfile);
		
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
		std::shared_ptr<Data::Spin_System> s1 = IO::Spin_System_from_Config(cfgfile);
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
		c = std::shared_ptr<Data::Spin_System_Chain>(new Data::Spin_System_Chain(sv, params_gneb, false));
		// Create transition of images
		Utility::Configuration_Chain::Homogeneous_Rotation(c, s1->spins, s7->spins);
		//-------------------------------------------------------------------------------
		

		//----------------------- LLG Iterations ----------------------------------------
		auto optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB());
		auto solver = new Engine::Solver_LLG(c, optim);
		// We could also do the following without using threads, but not without loss of generality
		s1->iteration_allowed = true;
		Utility::Threading::llg_threads[s1] = std::thread(&Engine::Solver_LLG::Iterate, solver);
		// To wait for started thread to finish, call:
		Threading::llg_threads[s1].join();
		//-------------------------------------------------------------------------------

	}
	catch (Exception ex) {
		if (ex == Exception::System_not_Initialized) {
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::IO, std::string("System not initialized - Terminating."));
		}
		else if (ex == Exception::Simulated_domain_too_small) {
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, std::string("CreateNeighbours:: Simulated domain is too small"));
		}
		else if (ex == Exception::Not_Implemented) {
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, std::string("Tried to use function which has not been implemented"));
		}
		else {
			Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, std::string("Unknown exception!"));
		}
	}

	// Finish
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= MonoSpin Finished =================");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
	Log.Append_to_File();
	return 0;
}