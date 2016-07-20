// emcc -std=c++11 main.cxx -o main.js -s EXPORTED_FUNCTIONS="['_test']"

// CC=emcc CXX=emcc cmake ..
// make
// mv libSpinEngine.a libSpinEngine.bc
// emcc libSpinEngine.bc -o testlib.js -s EXPORTED_FUNCTIONS="['_test']" -O2 -s ALLOW_MEMORY_GROWTH=1

#include <iostream>
//#include <cmath>
//#include <emscripten/emscripten.h>

#include "Version.h"
#include "Spin_System.h"
#include "Spin_System_Chain.h"
#include "Configurations.h"
#include "Configuration_Chain.h"

#include "Solver_LLG.h"
#include "Solver_GNEB.h"

#include "IO.h"
#include "Logging.h"

#include "Interface_State.h"

// Use Core Namespaces
using namespace Data;
using namespace Engine;
using namespace Utility;


Utility::LoggingHandler Utility::Log = Utility::LoggingHandler(Log_Level::WARNING, Log_Level::DEBUG, ".", "Log_" + Timing::CurrentDateTime() + ".txt");


extern "C" State * createSimulation()
{
  State *state = new State();
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "====================================================");
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=============== MonoSpin Initialising ==============");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= Version:  " + std::string(VERSION));
	Log.Send(Log_Level::INFO, Log_Sender::ALL, "================= Revision: " + std::string(VERSION_REVISION));
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "====================================================");

  //--- Read Log Levels
  IO::Log_Levels_from_Config("");

  //---------------------- initialize spin_systems --------------------------------
  // Create a system according to Config
  std::shared_ptr<Data::Spin_System> s1 = IO::Spin_System_from_Config("");
  //-------------------------------------------------------------------------------

  //---------------------- set images' configurations -----------------------------
  // Parameters
  double dir[3] = { 0,0,1 };
  std::vector<double> pos = { 14.5, 14.5, 0 };
  // First image is homogeneous with a Skyrmion at pos
  Configurations::Random(*s1);
  //Configurations::Skyrmion(*s1, pos, 6.0, 1.0, -90.0, false, false, false, false);
  //-------------------------------------------------------------------------------


  //----------------------- spin_system_chain -------------------------------------
  // Get parameters
  auto params_gneb = std::shared_ptr<Parameters_GNEB>(IO::GNEB_Parameters_from_Config(""));
  // Create the chain
  auto sv = std::vector<std::shared_ptr<Data::Spin_System>>();
  sv.push_back(s1);
  state->c = std::shared_ptr<Data::Spin_System_Chain>(new Data::Spin_System_Chain(sv, params_gneb, false));
  //-------------------------------------------------------------------------------

  // SIB optimizer
  state->optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB());
  
  // Allow iterations
  state->c->active_image = state->active_image;
  state->c->images[state->c->active_image]->iteration_allowed = true;
  state->c->iteration_allowed = false;

  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=============== MonoSpin Initialised ================");
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
  return state;
}

extern "C" double *getSpinDirections(State *state)
{
  // Return pointer to spins array
  double * result = (double *)state->c->images[state->c->active_image]->spins.data();
  return result;
}
  
extern "C" double *performIteration(State *state)
{
  // New Solver
  auto g = new Engine::Solver_LLG(state->c, state->optim);
  
  // n Iterations
  // for (int i=0; i<n; ++i) g->Iteration();
  g->Iteration();

  return getSpinDirections(state);
}
