/* This file solely exists to have a main function for emscripten. All
 * functionality used by the Web frontend should be part of the common library
 * C interface (see core/include/interface/).
 * The JS_LLG_Iteration function is specifically for JavaScript, where it
 * is (almost) impossible to have different threads for UI and simulation. 
 */
#include <memory>

#include "Method_LLG.h"
#include "Interface_State.h"

int main(void)
{
    return 0;
}


extern "C" void JS_LLG_Iteration(State *state)
{
    // LLG Method
    static std::shared_ptr<Engine::Method_LLG> method = std::shared_ptr<Engine::Method_LLG>(new Engine::Method_LLG(state->active_image, 0, 0));

    // SIB optimizer
    static std::shared_ptr<Engine::Optimizer> optimizer = std::shared_ptr<Engine::Optimizer>(new Engine::Optimizer_SIB(method));

    // Iterate
    optimizer->Iteration();
}
