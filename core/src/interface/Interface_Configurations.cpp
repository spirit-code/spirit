#include "Interface_Configurations.h"
#include "Interface_State.h"

#include "State.hpp"
#include "Configurations.hpp"


void Configuration_DomainWall(State *state, const double pos[3], double v[3], bool greater, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Apply configuration
    Utility::Configurations::DomainWall(*image, std::vector<double>(pos, pos+3), std::vector<double>(v, v+3), greater);
}

void Configuration_Homogeneous(State *state, double v[3], int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::Homogeneous(*image, std::vector<double>(v, v+3));
}

void Configuration_PlusZ(State *state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::PlusZ(*image);
}

void Configuration_MinusZ(State *state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::MinusZ(*image);
}

void Configuration_Random(State *state, bool external, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Apply configuration
    Utility::Configurations::Random(*image);
}

void Configuration_Add_Noise_Temperature(State *state, double temperature, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    Utility::Configurations::Add_Noise_Temperature(*image, temperature);
}

void Configuration_Hopfion(State *state, double pos[3], double r, int idx_image, int idx_chain)
{
	std::vector<double> position = { pos[0], pos[1], pos[2] };

	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Apply configuration
	Utility::Configurations::Hopfion(*image, position, r);
}

void Configuration_Skyrmion(State *state, double pos[3], double r, double order, double phase, bool upDown, bool achiral, bool rl, int idx_image, int idx_chain)
{
    std::vector<double> position = {pos[0], pos[1], pos[2]};

	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::Skyrmion(*image, position, r, order, phase, upDown, achiral, rl, false);
}

void Configuration_SpinSpiral(State *state, const char * direction_type, double q[3], double axis[3], double theta, int idx_image, int idx_chain)
{
    std::string dir_type(direction_type);

	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::SpinSpiral(*image, dir_type, std::vector<double>(q, q+3), std::vector<double>(axis, axis+3), theta);
}