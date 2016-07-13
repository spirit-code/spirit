#include "Interface_Parameters.h"
#include "Interface_State.h"

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set LLG
void Parameters_Set_LLG_Time_Step(State *state, float dt)
{
    auto s = state->c->images[state->c->active_image];
    auto p = s->llg_parameters;
    p->dt = dt;
}

void Parameters_Set_LLG_Damping(State *state, float damping)
{
    auto s = state->c->images[state->c->active_image];
    auto p = s->llg_parameters;
    p->damping = damping;
}

// Set GNEB
void Parameters_Set_GNEB_Spring_Constant(State *state, float spring_constant)
{
    auto p = state->c->gneb_parameters;
    p->spring_constant = spring_constant;
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get Parameters ---------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get LLG
void Parameters_Get_LLG_Time_Step(State *state, float * dt)
{
    auto s = state->c->images[state->c->active_image];
    auto p = s->llg_parameters;
    *dt = p->dt;

}

void Parameters_Get_LLG_Damping(State *state, float * damping)
{
    auto s = state->c->images[state->c->active_image];
    auto p = s->llg_parameters;
    *damping = p->damping;
}

// Set GNEB
void Parameters_Get_GNEB_Spring_Constant(State *state, float * spring_constant)
{
    auto p = state->c->gneb_parameters;
    *spring_constant = p->spring_constant;
}