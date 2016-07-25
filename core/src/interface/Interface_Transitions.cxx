#include "Interface_Transitions.h"
#include "Interface_State.h"
#include "Spin_System_Chain.h"
#include "Configuration_Chain.h"
#include <memory>

extern "C" void Transition_Homogeneous(State *state, int idx_chain)
{
    std::shared_ptr<Data::Spin_System_Chain> c;
    if (idx_chain < 0) c = state->active_chain;
    // Use this when State implements chain collection: else c = state->collection[idx_chain];
    Utility::Configuration_Chain::Homogeneous_Rotation(c, c->images[0]->spins, c->images[state->noi-1]->spins);
}