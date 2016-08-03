#include "Interface_Collection.h"
#include "Interface_State.h"

extern "C" int Collection_Get_NOC(State * state)
{
    return state->noc;
}

extern "C" void Collection_next_Chain(State * state)
{

}

extern "C" void Collection_prev_Chain(State * state)
{

}