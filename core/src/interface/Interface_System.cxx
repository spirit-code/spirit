#include "Interface_System.h"
#include "Interface_State.h"

extern "C" int System_Get_Index(State * state)
{
    return state->idx_active_image;
}

extern "C" int System_Get_NOS(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return image->nos;
}

extern "C" double * System_Get_Spin_Directions(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return (double *)image->spins->data();
}