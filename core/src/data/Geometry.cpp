#include <Geometry.hpp>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
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

  const std::vector<tetrahedron_t>& Geometry::triangulation() {
    if (is2D()) {
      _triangulation.clear();
      return _triangulation;
    }
    if (_triangulation.size() == 0) {
      bool is_simple_regular_geometry = true;
      if (is_simple_regular_geometry) {
        _triangulation.clear();
        int cell_indices[] = {
          0, 1, 5, 3,
          1, 3, 2, 5,
          3, 2, 5, 6,
          7, 6, 5, 3,
          4, 7, 5, 3,
          0, 4, 3, 5
        };
        int x_offset = 1;
        int y_offset = n_cells[0];
        int z_offset = n_cells[0]*n_cells[1];
        int offsets[] = {
          0, x_offset, x_offset+y_offset, y_offset,
          z_offset, x_offset+z_offset, x_offset+y_offset+z_offset, y_offset+z_offset
        };
        
        for (int ix = 0; ix < n_cells[0]-1; ix++) {
          for (int iy = 0; iy < n_cells[1]-1; iy++) {
            for (int iz = 0; iz < n_cells[0]-1; iz++) {
              int base_index = ix*x_offset+iy*y_offset+iz*z_offset;
              for (int j = 0; j < 6; j++) {
                tetrahedron_t tetrahedron;
                for (int k = 0; k < 4; k++) {
                  int index = base_index + offsets[cell_indices[j*4+k]];
                  tetrahedron.point_indices[k] = index;
                }
                _triangulation.push_back(tetrahedron);
              }
            }
          }
        }
      } else {
        std::vector<vector_t> points;
        points.resize(spin_pos.size()/3);
        for (std::vector<vector_t>::size_type i = 0; i < points.size(); i++) {
          points[i].x = spin_pos[i];
          points[i].y = spin_pos[points.size()+i];
          points[i].z = spin_pos[points.size()*2+i];
        }
        _triangulation = compute_delaunay_triangulation(points);
      }
    }
    return _triangulation;
  }
  bool Geometry::is2D() const {
    return (n_spins_basic_domain == 1) && (n_cells[0] == 1 || n_cells[1] == 1 || n_cells[2] == 1);
  }
}

