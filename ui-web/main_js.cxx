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


// Use Core Namespaces
using namespace Data;
using namespace Engine;
using namespace Utility;

std::shared_ptr<Data::Spin_System_Chain> c;
Utility::LoggingHandler Utility::Log = Utility::LoggingHandler(Log_Level::WARNING, Log_Level::DEBUG, ".", "Log_" + Timing::CurrentDateTime() + ".txt");
std::shared_ptr<Engine::Optimizer> optim;

void init()
{
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "====================================================");
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=============== MonoSpin Initialising ==============");
	Log.Send(Log_Level::ALL, Log_Sender::ALL, "================= Version:  " + std::string(VERSION));
	Log.Send(Log_Level::INFO, Log_Sender::ALL, "================= Revision:" + std::string(VERSION_REVISION));
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
  c = std::shared_ptr<Data::Spin_System_Chain>(new Data::Spin_System_Chain(sv, params_gneb, false));
  //-------------------------------------------------------------------------------

  // SIB optimizer
  optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB());
  
  // Allow iterations
  c->images[0]->iteration_allowed = true;
  c->iteration_allowed = false;

  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=============== MonoSpin Initialised ================");
  Log.Send(Log_Level::ALL, Log_Sender::ALL, "=====================================================");

}
  
extern "C" double *test(int n)
{
  // If not yet initialized, call init
  static bool is_init = false;
  if (!is_init)
  {
    init();
    is_init = true;
  }
  
  // New Solver
  auto g = new Engine::Solver_LLG(c, optim);
  
  // n Iterations
  // for (int i=0; i<n; ++i) g->Iteration();
  g->Iteration();

  // Return pointer to spins array
  double * result = (double *)c->images[0]->spins.data();
  return result;
}

int main(int argc, char ** argv)
{
  test(1);
}