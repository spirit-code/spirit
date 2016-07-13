#pragma once
#ifndef INTERFACE_GEOMETRY_H
#define INTERFACE_GEOMETRY_H
struct State;

extern "C" void Geometry_Get_Bounds(State *state, float * xmin, float * xmax, float * ymin, float * ymax, float * zmin, float * zmax);

#endif