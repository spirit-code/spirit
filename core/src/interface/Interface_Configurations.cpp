#include "Interface_Configurations.h"
#include "Interface_State.h"

#include "Core_Defines.h"
#include "State.hpp"
#include "Configurations.hpp"


void Configuration_DomainWall(State *state, const float pos[3], float v[3], bool greater, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Apply configuration
	Utility::Configurations::DomainWall(*image, Vector3{ pos[0],pos[1],pos[2] }, Vector3{ v[0],v[1],v[2] }, greater);
}

void Configuration_Homogeneous(State *state, float v[3], int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::Homogeneous(*image, Vector3{ v[0],v[1],v[2] });
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

void Configuration_Add_Noise_Temperature(State *state, float temperature, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    Utility::Configurations::Add_Noise_Temperature(*image, temperature);
}

void Configuration_Hopfion(State *state, float pos[3], float r, int order, int idx_image, int idx_chain)
{
	Vector3 position = { pos[0], pos[1], pos[2] };

	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	// Apply configuration
	Utility::Configurations::Hopfion(*image, position, r, order);
}

void Configuration_Skyrmion(State *state, float pos[3], float r, float order, float phase, bool upDown, bool achiral, bool rl, int idx_image, int idx_chain)
{
    Vector3 position = {pos[0], pos[1], pos[2]};

	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
    Utility::Configurations::Skyrmion(*image, position, r, order, phase, upDown, achiral, rl, false);
}

void Configuration_SpinSpiral(State *state, const char * direction_type, float q[3], float axis[3], float theta, int idx_image, int idx_chain)
{
    std::string dir_type(direction_type);

	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Apply configuration
	Utility::Configurations::SpinSpiral(*image, dir_type, Vector3{ q[0], q[1], q[2] }, Vector3{ axis[0], axis[1], axis[2] }, theta);
}