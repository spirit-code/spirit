#include <interface/Interface_Transitions.h>
#include <interface/Interface_State.h>
#include <data/State.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/Configuration_Chain.hpp>

#include <memory>

void Transition_Homogeneous(State *state, int idx_1, int idx_2, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Use this when State implements chain collection: else c = state->collection[idx_chain];
    Utility::Configuration_Chain::Homogeneous_Rotation(chain, idx_1, idx_2);
}

void Transition_Add_Noise_Temperature(State *state, float temperature, int idx_1, int idx_2, int idx_chain)
{
	int idx_image = -1;
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    // Use this when State implements chain collection: else c = state->collection[idx_chain];
    Utility::Configuration_Chain::Add_Noise_Temperature(chain, idx_1, idx_2, temperature);
}