#include "Configurations.h"
#include "Interface_Globals.h"
#include "Interface_Configurations.h"

void DomainWall(const double pos[3], double v[3], bool greater)
{
    Utility::Configurations::DomainWall(*c->images[c->active_image], pos, v, greater);
}

void Homogeneous(double v[3])
{
    Utility::Configurations::Homogeneous(*c->images[c->active_image], v);
}

void PlusZ()
{
    Utility::Configurations::PlusZ(*c->images[c->active_image]);
}

void MinusZ()
{
    Utility::Configurations::MinusZ(*c->images[c->active_image]);
}

void Random(bool external)
{
    Utility::Configurations::Random(*c->images[c->active_image]);
}

void Skyrmion(std::vector<double> pos, double r, double speed, double order, bool upDown, bool achiral, bool rl, bool experimental)
{
    Utility::Configurations::Skyrmion(*c->images[c->active_image], pos, r, speed, order, upDown, achiral, rl, experimental);
}

void SpinSpiral(std::string direction_type, double q[3], double axis[3], double theta)
{
    Utility::Configurations::SpinSpiral(*c->images[c->active_image], direction_type, q, axis, theta);
}