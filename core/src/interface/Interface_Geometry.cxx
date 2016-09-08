#include "Interface_Geometry.h"
#include "Interface_State.h"

extern "C" double * Geometry_Get_Spin_Positions(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return (double *)image->geometry->spin_pos.data();
}

extern "C" void Geometry_Get_Bounds(State *state, float * min, float * max, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    for (int dim=0; dim<3; ++dim)
    {
        min[dim] = (float)g->bounds_min[dim];
        max[dim] = (float)g->bounds_max[dim];
    }   
}

// Get Center as array (x,y,z)
extern "C" void Geometry_Get_Center(State *state, float * center, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    for (int dim=0; dim<3; ++dim)
    {
        center[dim] = (float)g->center[dim];
    }
}

// Get basis vectors ta, tb, tc
extern "C" void Geometry_Get_Basis_Vectors(State *state, float * a, float * b, float * c, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    for (int dim=0; dim<3; ++dim)
    {
        a[dim] = (float)g->basis[dim][0];
        b[dim] = (float)g->basis[dim][1];
        c[dim] = (float)g->basis[dim][2];
    }
}

// TODO: Get basis atoms
// extern "C" void Geometry_Get_Basis_Atoms(State *state, float * n_atoms, float ** atoms)
// {
//     auto g = state->active_image->geometry;
//     *n_atoms = g->n_spins_basic_domain;
// }

// Get number of basis cells in the three translation directions
extern "C" void Geometry_Get_N_Cells(State *state, int * na, int * nb, int * nc, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    *na = g->n_cells[0];
    *nb = g->n_cells[1];
    *nc = g->n_cells[2];
}
// Get translation vectors ta, tb, tc
extern "C" void Geometry_Get_Translation_Vectors(State *state, float * ta, float * tb, float * tc, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    for (int dim=0; dim<3; ++dim)
    {
        ta[dim] = (float)g->translation_vectors[dim][0];
        tb[dim] = (float)g->translation_vectors[dim][1];
        tc[dim] = (float)g->translation_vectors[dim][2];
    }
}

extern "C" bool Geometry_Is_2D(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    bool is2D = true;
    auto g = image->geometry;
    for (int i=0; i<g->n_spins_basic_domain; ++i)
    {
        // We assume that the z-component of a 2D system will always be zero
        if (g->spin_pos[2*g->nos+i] != 0) is2D = false;
    }
    return is2D;
}