#include <interface/Interface_System.h>
#include <interface/Interface_State.h>
#include <data/State.hpp>

int System_Get_Index(State * state)
{
    return state->idx_active_image;
}

int System_Get_NOS(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return image->nos;
}

scalar * System_Get_Spin_Directions(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);
	
    return (scalar *)(*image->spins)[0].data();
}

scalar * System_Get_Effective_Field(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	return image->effective_field[0].data();
}

float System_Get_Rx(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	return (float)chain->Rx[idx_image];
}

float System_Get_Energy(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return (float)image->E;
}

void System_Get_Energy_Array(State * state, float * energies, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    for (int i=0; i<image->E_array.size(); ++i)
    {
        energies[i] = (float)image->E_array[i].second;
    }
}

void System_Update_Data(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    image->UpdateEnergy();
}