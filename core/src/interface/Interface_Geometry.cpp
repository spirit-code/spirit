#include "Interface_Geometry.h"
#include "Interface_State.h"

#include "State.hpp"

double * Geometry_Get_Spin_Positions(State * state, int idx_image, int idx_chain)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices(state, idx_image, idx_chain, image, chain);

    return (double *)image->geometry->spin_pos.data();
}

void Geometry_Get_Bounds(State *state, float * min, float * max, int idx_image, int idx_chain)
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
void Geometry_Get_Center(State *state, float * center, int idx_image, int idx_chain)
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
void Geometry_Get_Basis_Vectors(State *state, float * a, float * b, float * c, int idx_image, int idx_chain)
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

// Get number of basis cells in the three translation directions
void Geometry_Get_N_Cells(State *state, int * n_cells, int idx_image, int idx_chain)
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
void Geometry_Get_Translation_Vectors(State *state, float * ta, float * tb, float * tc, int idx_image, int idx_chain)
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

bool Geometry_Is_2D(State * state, int idx_image, int idx_chain)
{
	std::shared_ptr<Data::Spin_System> image;
	std::shared_ptr<Data::Spin_System_Chain> chain;
	from_indices(state, idx_image, idx_chain, image, chain);

  auto g = image->geometry;
  return g->is2D();
}


int Geometry_Get_Triangulation(State * state, int **indices_ptr, int idx_image, int idx_chain)
{
  std::shared_ptr<Data::Spin_System> image;
  std::shared_ptr<Data::Spin_System_Chain> chain;
  from_indices(state, idx_image, idx_chain, image, chain);

  auto g = image->geometry;
  auto tetrahedra = g->triangulation();
  if (indices_ptr != nullptr) {
    *indices_ptr = &tetrahedra.data()[0].point_indices[0];
	*indices_ptr = new int[tetrahedra.size() * 4];
	for (int i = 0; i < tetrahedra.size(); ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			(*indices_ptr)[4 * i + j] = tetrahedra[i].point_indices[j];
		}
	}
  }
  return tetrahedra.size();
}
