#include "Configurations.h"
#include "Interface_State.h"
#include "Interface_Configurations.h"

void Configuration_DomainWall(State *state, const double pos[3], double v[3], bool greater, int idx_image, int idx_chain)
{
    Utility::Configurations::DomainWall(*state->active_image, std::vector<double>(pos, pos+3), std::vector<double>(v, v+3), greater);
}

void Configuration_Homogeneous(State *state, double v[3], int idx_image, int idx_chain)
{
    // Get the image
    std::shared_ptr<Data::Spin_System_Chain> c;
    if (idx_chain < 0) c = state->active_chain;
    std::shared_ptr<Data::Spin_System> img;
    if (idx_image < 0) img = state->active_image;
    else img = c->images[idx_image];
    // Apply configuration
    Utility::Configurations::Homogeneous(*img, std::vector<double>(v, v+3));
}

void Configuration_PlusZ(State *state, int idx_image, int idx_chain)
{
    // Get the image
    std::shared_ptr<Data::Spin_System_Chain> c;
    if (idx_chain < 0) c = state->active_chain;
    std::shared_ptr<Data::Spin_System> img;
    if (idx_image < 0) img = state->active_image;
    else img = c->images[idx_image];
    // Apply configuration
    Utility::Configurations::PlusZ(*img);
}

void Configuration_MinusZ(State *state, int idx_image, int idx_chain)
{
    // Get the image
    std::shared_ptr<Data::Spin_System_Chain> c;
    if (idx_chain < 0) c = state->active_chain;
    std::shared_ptr<Data::Spin_System> img;
    if (idx_image < 0) img = state->active_image;
    else img = c->images[idx_image];
    // Apply configuration
    Utility::Configurations::MinusZ(*img);
}

void Configuration_Random(State *state, bool external, int idx_image, int idx_chain)
{
    Utility::Configurations::Random(*state->active_image);
}

void Configuration_Add_Noise_Temperature(State *state, double temperature, int idx_image, int idx_chain)
{
    Utility::Configurations::Add_Noise_Temperature(*state->active_image, temperature);
}

void Configuration_Skyrmion(State *state, double pos[3], double r, double order, double phase, bool upDown, bool achiral, bool rl, int idx_image, int idx_chain)
{
    std::vector<double> position = {pos[0], pos[1], pos[2]};
    // Get the image
    std::shared_ptr<Data::Spin_System_Chain> c;
    if (idx_chain < 0) c = state->active_chain;
    std::shared_ptr<Data::Spin_System> img;
    if (idx_image < 0) img = state->active_image;
    else img = c->images[idx_image];
    // Apply configuration
    Utility::Configurations::Skyrmion(*img, position, r, order, phase, upDown, achiral, rl, false);
}

void Configuration_SpinSpiral(State *state, const char * direction_type, double q[3], double axis[3], double theta, int idx_image, int idx_chain)
{
    std::string dir_type(direction_type);
    // Get the image
    std::shared_ptr<Data::Spin_System_Chain> c;
    if (idx_chain < 0) c = state->active_chain;
    std::shared_ptr<Data::Spin_System> img;
    if (idx_image < 0) img = state->active_image;
    else img = c->images[idx_image];
    // Apply configuration
    Utility::Configurations::SpinSpiral(*img, dir_type, std::vector<double>(q, q+3), std::vector<double>(axis, axis+3), theta);
}