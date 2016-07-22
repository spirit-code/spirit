#include "Interface_Geometry.h"
#include "Interface_State.h"

void Geometry_Get_Bounds(State *state, float * xmin, float * ymin, float * zmin, float * xmax, float * ymax, float * zmax)
{
    auto g = state->c->images[state->c->active_image]->geometry;
    // Min
    *xmin = g->bounds_min[0];
    *ymin = g->bounds_min[1];
    *zmin = g->bounds_min[2];
    // Max
    *xmax = g->bounds_max[0];
    *ymax = g->bounds_max[1];
    *zmax = g->bounds_max[2];
}