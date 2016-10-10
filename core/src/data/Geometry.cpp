#include <Geometry.h>
#include "Qhull.h"
#include "QhullFacetList.h"
#include "QhullVertexSet.h"

namespace Data
{
	Geometry::Geometry(std::vector<std::vector<double>> basis_i, const std::vector<std::vector<double>> translation_vectors_i,
		const std::vector<int> n_cells_i, const int n_spins_basic_domain_i, const std::vector<double> spin_pos_i) :
		basis(basis_i), translation_vectors(translation_vectors_i),
		n_cells(n_cells_i), n_spins_basic_domain(n_spins_basic_domain_i),
		spin_pos(spin_pos_i), nos( n_spins_basic_domain_i * n_cells_i[0] * n_cells_i[1] * n_cells_i[2])
	{
		// Calculate Bounds of the System
		this->bounds_min = std::vector<double>(3);
		this->bounds_max = std::vector<double>(3);
		this->center = std::vector<double>(3);

		for (int dim = 0; dim < 3; ++dim)
		{
			for (int iatom = 0; iatom < nos; ++iatom)
			{
				if (this->spin_pos[dim*nos + iatom] < this->bounds_min[dim]) this->bounds_min[dim] = spin_pos[dim*nos + iatom];
				if (this->spin_pos[dim*nos + iatom] > this->bounds_max[dim]) this->bounds_max[dim] = spin_pos[dim*nos + iatom];
			}
		}

		// Calculate Center of the System
		for (int dim = 0; dim < 3; ++dim)
		{
			this->center[dim] = (this->bounds_min[dim] + this->bounds_max[dim]) / 2.0;
		}
	}

    std::vector<tetrahedron_t> compute_delaunay_triangulation(const std::vector<vector_t> &points) {
        const int ndim = 3;
        std::vector<tetrahedron_t> tetrahedra;
        tetrahedron_t tmp_tetrahedron;
        int *current_index;

        orgQhull::Qhull qhull;
        qhull.runQhull("", ndim, points.size(), (coordT *) points.data(),  "qhull d Qt Qbb Qz");

        orgQhull::QhullFacetList facet_list = qhull.facetList();
        for(orgQhull::QhullFacetList::iterator facet_it = facet_list.begin(); facet_it != facet_list.end(); ++facet_it) {
            if(!facet_it->isUpperDelaunay()) {
                orgQhull::QhullVertexSet vertices = facet_it->vertices();
                current_index = &tmp_tetrahedron.point_indices[0];
                for(orgQhull::QhullVertexSet::iterator vertex_it = vertices.begin(); vertex_it != vertices.end(); ++vertex_it) {
                    *current_index++ = (*vertex_it).point().id();
                }
                tetrahedra.push_back(tmp_tetrahedron);
            }
        }

        return tetrahedra;
    }
}

