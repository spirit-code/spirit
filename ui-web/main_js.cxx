// emcc -std=c++11 main.cxx -o main.js -s EXPORTED_FUNCTIONS="['_test']"

// CC=emcc CXX=emcc cmake ..
// make
// mv libSpinEngine.a libSpinEngine.bc
// emcc libSpinEngine.bc -o testlib.js -s EXPORTED_FUNCTIONS="['_test']" -O2 -s ALLOW_MEMORY_GROWTH=1

#include <iostream>
//#include <cmath>
//#include <emscripten/emscripten.h>

#include "Spin_System.h"
#include "Spin_System_Chain.h"

#include "Logging.h"

#include "Interface_State.h"

// Use Core Namespaces
using namespace Data;
using namespace Engine;
using namespace Utility;

// Create a Log
Utility::LoggingHandler Utility::Log = Utility::LoggingHandler(Log_Level::WARNING, Log_Level::DEBUG, ".", "Log_" + Timing::CurrentDateTime() + ".txt");

// Handle the Play/Pause button
extern "C" void PlayPause(State *state)
{
    // LLG Simulations
    if (true)
    {
        // Test if already running
        if (state->c->images[state->idx_active_image]->iteration_allowed || state->c->iteration_allowed)
        {
        state->c->images[state->idx_active_image]->iteration_allowed = false;
        }
        else
        {
            // Allow iterations
            state->c->images[state->idx_active_image]->iteration_allowed = true;
            state->c->iteration_allowed = false;
            // SIB optimizer
            auto optim = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB());
            // New Solver
            state->solvers_llg[state->idx_active_chain][state->idx_active_image] = new Engine::Solver_LLG(state->c, optim);
            // Iterate
            state->solvers_llg[state->idx_active_chain][state->idx_active_image]->Iterate();
        }
    }
    // GNEB Calculations
    else if (false)
    {
        // not yet implemented
    }
    // MMF Calculations
    else if (false)
    {
        // not yet implemented
    }
}

// Get the double pointer to the spins array
extern "C" double *getSpinDirections(State *state)
{
    // Return pointer to spins array
    double * result = (double *)state->c->images[state->c->active_image]->spins.data();
    return result;
}