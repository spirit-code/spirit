/* This file solely exists to have a main function for emscripten. All
 * functionality used by the Web frontend should be part of the common library
 * C interface (see core/include/interface/).
 * The JS_LLG_Iteration function is specifically for JavaScript, where it
 * is (almost) impossible to have different threads for UI and simulation. 
 */
#include <memory>

#include "Spirit/State.h"

// TODO: remove these and use SingleShot API function instead
#include "State.hpp"
#include "Method_LLG.hpp"
/////

int main(void)
{
    return 0;
}


extern "C" void JS_LLG_Iteration(State *state)
{
    // LLG Method
    static std::shared_ptr<Engine::Method_LLG> method =
        std::shared_ptr<Engine::Method>( new Engine::Method_LLG<Engine::Solver::SIB>( state->active_image ) );

    // Iterate
    method->Iteration();
}
