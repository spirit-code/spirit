#include "Configurations.h"
#include "Interface_State.h"
#include "Interface_Configurations.h"

void Configuration_DomainWall(State *state, const double pos[3], double v[3], bool greater)
{
    Utility::Configurations::DomainWall(*state->active_image, pos, v, greater);
}

void Configuration_Homogeneous(State *state, double v[3])
{
    Utility::Configurations::Homogeneous(*state->active_image, v);
}

void Configuration_PlusZ(State *state)
{
    Utility::Configurations::PlusZ(*state->active_image);
}

void Configuration_MinusZ(State *state)
{
    Utility::Configurations::MinusZ(*state->active_image);
}

void Configuration_Random(State *state, bool external)
{
    Utility::Configurations::Random(*state->active_image);
}

void Configuration_Skyrmion(State *state, double pos[3], double r, double order, double phase, bool upDown, bool achiral, bool rl, bool experimental)
{
    std::vector<double> position = {pos[0], pos[1], pos[2]};
    Utility::Configurations::Skyrmion(*state->active_image, position, r, order, phase, upDown, achiral, rl, experimental);
}

void Configuration_SpinSpiral(State *state, const char * direction_type, double q[3], double axis[3], double theta)
{
    std::string dir_type(direction_type);
    Utility::Configurations::SpinSpiral(*state->active_image, dir_type, q, axis, theta);
}