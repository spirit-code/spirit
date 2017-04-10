#include <Spirit/Geometry.h>
#include <Spirit/State.h>

#include <data/State.hpp>

scalar * Geometry_Get_Spin_Positions(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return (scalar *)image->geometry->spin_pos[0].data();
}

void Geometry_Get_Bounds(State *state, float min[3], float max[3], int idx_image, int idx_chain)
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
void Geometry_Get_Center(State *state, float center[3], int idx_image, int idx_chain)
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

void Geometry_Get_Cell_Bounds(State *state, float min[3], float max[3], int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    for (int dim=0; dim<3; ++dim)
    {
        min[dim] = (float)g->cell_bounds_min[dim];
        max[dim] = (float)g->cell_bounds_max[dim];
    }   
}

// Get basis vectors ta, tb, tc
void Geometry_Get_Basis_Vectors(State *state, float a[3], float b[3], float c[3], int idx_image, int idx_chain)
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
// void Geometry_Get_Basis_Atoms(State *state, float * n_atoms, float ** atoms)
// {
//     auto g = state->active_image->geometry;
//     *n_atoms = g->n_spins_basic_domain;
// }

// Get number of atoms in a basis cell
int Geometry_Get_N_Basis_Atoms(State *state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    return image->geometry->n_spins_basic_domain;
}

// Get number of basis cells in the three translation directions
void Geometry_Get_N_Cells(State *state, int n_cells[3], int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    n_cells[0] = g->n_cells[0];
	n_cells[1] = g->n_cells[1];
	n_cells[2] = g->n_cells[2];
}
// Get translation vectors ta, tb, tc
void Geometry_Get_Translation_Vectors(State *state, float ta[3], float tb[3], float tc[3], int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

    auto g = image->geometry;
    for (int dim=0; dim<3; ++dim)
    {
        ta[dim] = (float)g->translation_vectors[0][dim];
        tb[dim] = (float)g->translation_vectors[1][dim];
        tc[dim] = (float)g->translation_vectors[2][dim];
    }
}

int Geometry_Get_Dimensionality(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

	auto g = image->geometry;
	return g->dimensionality;
}


int Geometry_Get_Triangulation(State * state, const int ** indices_ptr, int n_cell_step, int idx_image, int idx_chain)
{
  std::shared_ptr<Data::Spin_System> image;
  std::shared_ptr<Data::Spin_System_Chain> chain;
  from_indices(state, idx_image, idx_chain, image, chain);

  auto g = image->geometry;
  auto& tetrahedra = g->triangulation(n_cell_step);
  if (indices_ptr != nullptr) {
	  *indices_ptr = reinterpret_cast<const int *>(tetrahedra.data());
  }
  return tetrahedra.size();
}
