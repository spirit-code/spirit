#include "Interface_System.h"
#include "Interface_State.h"

extern "C" int System_Get_Index(State * state)
{
    return state->idx_active_image;
}

extern "C" int System_Get_NOS(State * state)
{
    return state->active_image->nos;
}

// void System_Get_Spin_Directions(State *state, int idx_image, int idx_chain)
// {
    
// }