#pragma once
#ifndef INTERFACE_GEOMETRY_H
#define INTERFACE_GEOMETRY_H
struct State;

extern "C" void Geometry_Get_Bounds(State *state, float * xmin, float * ymin, float * zmin, float * xmax, float * ymax, float * zmax);
// TODO: get spin positions

#endif