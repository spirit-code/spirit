#include "Threading.h"
#include "Signal.h"

#include "Interface_State.h"
#include "Interface_Chain.h"
#include "Interface_Configurations.h"
#include "Interface_Transitions.h"

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

	//--- Register SigInt
	signal(SIGINT, Signal::Handle_SigInt);
	
	//---------------------- file names ---------------------------------------------
	//--- Config Files
	// const char * cfgfile = "input/markus-paper.cfg";
	const char * cfgfile = "input/gideon-master-thesis-isotropic.cfg";
	// const char * cfgfile = "input/daniel-master-thesis-isotropic.cfg";
	//--- Data Files
	// std::string spinsfile = "input/anisotropic/achiral.txt";
	// std::string chainfile = "input/chain.txt";
	//-------------------------------------------------------------------------------
	
	//--- Initialise State
	state = std::shared_ptr<State>(setupState(cfgfile));
//---------------------- initialize spin_systems --------------------------------
	// Copy the system a few times
	auto sc = Spin_System(*state->active_image);
	for (int i=1; i<7; ++i)
	{
		Chain_Insert_Image_After(state.get(), sc);
	}
	//-------------------------------------------------------------------------------
	
	//----------------------- spin_system_chain -------------------------------------
	// Parameters
	double dir[3] = { 0,0,1 };
	double pos[3] = { 14.5, 14.5, 0 };

	// Read Image from file
	//Configuration_from_File(state.get(), spinsfile, 0);
	// Read Chain from file
	//Chain_from_File(state.get(), chainfile);

	// First image is homogeneous with a Skyrmion at pos
	Configuration_Homogeneous(state.get(), dir, 0);
	Configuration_Skyrmion(state.get(), pos, 6.0, 1.0, -90.0, false, false, false, 0);
	// Last image is homogeneous
	Configuration_Homogeneous(state.get(), dir, state->noi-1);

	// Create transition of images between first and last
	Transition_Homogeneous(state.get(), 0, state->noi-1);
	//-------------------------------------------------------------------------------

	//----------------------- LLG Iterations ----------------------------------------
	auto optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB());
	auto solver = new Engine::Solver_LLG(state->active_chain, optim);
	// We could also do the following without using threads, but not without loss of generality
	state->active_image->iteration_allowed = true;
	Utility::Threading::llg_threads[state->active_image] = std::thread(&Engine::Solver_LLG::Iterate, solver);
	// To wait for started thread to finish, call:
	Threading::llg_threads[state->active_image].join();
	//-------------------------------------------------------------------------------


	// Finish
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= MonoSpin Finished =================");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
	Log.Append_to_File();
	return 0;
}