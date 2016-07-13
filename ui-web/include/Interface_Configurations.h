#pragma once
#ifndef INTERFACE_CONFIGURATIONS_H
#define INTERFACE_CONFIGURATIONS_H
struct State;
// orients all spins with x>pos into the direction of the v
extern "C" void DomainWall(State *state, const double pos[3], double v[3], const bool greater = true);

// points all Spins parallel to the direction of v
// calls DomainWall (s, -1E+20, v)
extern "C" void Homogeneous(State *state, double v[3]);
// points all Spins in +z direction
extern "C" void PlusZ(State *state);
// points all Spins in -z direction
extern "C" void MinusZ(State *state);

// points all Spins in random directions
extern "C" void Random(State *state, bool external = false);

// points a sperical region of spins of radius r
// into direction of vec at position pos
extern "C" void Skyrmion(State *state, std::vector<double> pos, double r, double speed, double order, bool upDown, bool achiral, bool rl, bool experimental);
// Spin Spiral
extern "C" void SpinSpiral(State *state, std::string direction_type, double q[3], double axis[3], double theta);

#endif