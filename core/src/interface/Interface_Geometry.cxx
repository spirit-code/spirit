#include "Interface_Geometry.h"
#include "Interface_State.h"

void Geometry_Get_Bounds(State *state, float * xmin, float * ymin, float * zmin, float * xmax, float * ymax, float * zmax)
{
    auto g = state->active_image->geometry;
    // Min
    *xmin = (float)g->bounds_min[0];
    *ymin = (float)g->bounds_min[1];
    *zmin = (float)g->bounds_min[2];
    // Max
    *xmax = (float)g->bounds_max[0];
    *ymax = (float)g->bounds_max[1];
    *zmax = (float)g->bounds_max[2];
}