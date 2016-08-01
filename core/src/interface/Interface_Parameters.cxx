#include "Interface_Parameters.h"
#include "Interface_State.h"

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set LLG
void Parameters_Set_LLG_Time_Step(State *state, float dt)
{
    auto s = state->active_image;
    auto p = s->llg_parameters;
    p->dt = dt;
}

void Parameters_Set_LLG_Damping(State *state, float damping)
{
    auto s = state->active_image;
    auto p = s->llg_parameters;
    p->damping = damping;
}

// Set GNEB
void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant)
{
    auto p = state->active_chain->gneb_parameters;
    p->spring_constant = spring_constant;
}

void Parameters_Set_GNEB_Climbing_Falling(State *state, bool climbing, bool falling)
{
    state->active_chain->climbing_image[state->idx_active_image] = climbing;
    state->active_chain->falling_image[state->idx_active_image] = falling;
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get LLG
void Parameters_Get_LLG_Time_Step(State *state, float * dt)
{
    auto s = state->active_image;
    auto p = s->llg_parameters;
    *dt = p->dt;

}

void Parameters_Get_LLG_Damping(State *state, float * damping)
{
    auto s = state->active_image;
    auto p = s->llg_parameters;
    *damping = p->damping;
}

// Set GNEB
void Parameters_Get_GNEB_Spring_Constant(State *state, float * spring_constant)
{
    auto p = state->active_chain->gneb_parameters;
    *spring_constant = p->spring_constant;
}

void Parameters_Get_GNEB_Climbing_Falling(State *state, bool * climbing, bool * falling)
{
    *climbing = state->active_chain->climbing_image[state->idx_active_image];
    *falling = state->active_chain->falling_image[state->idx_active_image];
}