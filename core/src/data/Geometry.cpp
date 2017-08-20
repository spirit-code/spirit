#include <data/Geometry.hpp>
#include <engine/Neighbours.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Qhull.h"
#include "QhullFacetList.h"
#include "QhullVertexSet.h"

#include <array>

namespace Data
{
    Geometry::Geometry(std::vector<Vector3> basis_i, const std::vector<Vector3> translation_vectors_i,
        const intfield n_cells_i, const std::vector<Vector3> basis_atoms_i, const scalar lattice_constant_i,
        const vectorfield spin_pos_i, const intfield atom_types) :
        basis(basis_i), translation_vectors(translation_vectors_i), n_cells(n_cells_i),
        n_spins_basic_domain(basis_atoms_i.size()), basis_atoms(basis_atoms_i), lattice_constant(lattice_constant_i),
        spin_pos(spin_pos_i), nos(basis_atoms_i.size() * n_cells_i[0] * n_cells_i[1] * n_cells_i[2]), atom_types(atom_types)
    {
        // Calculate Bounds of the System
        this->bounds_max.setZero();
        this->bounds_min.setZero();
        for (int iatom = 0; iatom < nos; ++iatom)
        {
            for (int dim = 0; dim < 3; ++dim)
            {
                if (this->spin_pos[iatom][dim] < this->bounds_min[dim]) this->bounds_min[dim] = spin_pos[iatom][dim];
                if (this->spin_pos[iatom][dim] > this->bounds_max[dim]) this->bounds_max[dim] = spin_pos[iatom][dim];
            }
        }

        // Calculate Unit Cell Bounds
        this->cell_bounds_max.setZero();
        this->cell_bounds_min.setZero();
        for (unsigned int ivec = 0; ivec < translation_vectors.size(); ++ivec)
        {
            for (int iatom = 0; iatom < n_spins_basic_domain; ++iatom)
            {
                auto neighbour1 = basis_atoms[iatom] + translation_vectors[ivec];
                auto neighbour2 = basis_atoms[iatom] - translation_vectors[ivec];
                for (int dim = 0; dim < 3; ++dim)
                {
                    if (neighbour1[dim] < this->cell_bounds_min[dim]) this->cell_bounds_min[dim] = neighbour1[dim];
                    if (neighbour1[dim] > this->cell_bounds_max[dim]) this->cell_bounds_max[dim] = neighbour1[dim];
                    if (neighbour2[dim] < this->cell_bounds_min[dim]) this->cell_bounds_min[dim] = neighbour2[dim];
                    if (neighbour2[dim] > this->cell_bounds_max[dim]) this->cell_bounds_max[dim] = neighbour2[dim];
                }
            }
        }
        this->cell_bounds_min *= 0.5;
        this->cell_bounds_max *= 0.5;

        // Calculate Center of the System
        for (int dim = 0; dim < 3; ++dim)
        {
            this->center[dim] = (this->bounds_min[dim] + this->bounds_max[dim]) / 2.0;
        }

        // Calculate dimensionality
        this->dimensionality = calculateDimensionality();

        // For updates of triangulation and tetrahedra
        this->last_update_n_cell_step = -1;
        this->last_update_n_cells = intfield(3, -1);
    }

    std::vector<tetrahedron_t> compute_delaunay_triangulation_3D(const std::vector<vector3_t> & points)
    {
        const int ndim = 3;
        std::vector<tetrahedron_t> tetrahedra;
        tetrahedron_t tmp_tetrahedron;
        int *current_index;

        orgQhull::Qhull qhull;
        qhull.runQhull("", ndim, points.size(), (coordT *) points.data(),  "d Qt Qbb Qz");
        orgQhull::QhullFacetList facet_list = qhull.facetList();
        for(const auto & facet : facet_list)
        {
            if(!facet.isUpperDelaunay())
            {
                current_index = &tmp_tetrahedron[0];
                for(const auto & vertex : facet.vertices())
                {
                    *current_index++ = vertex.point().id();
                }
                tetrahedra.push_back(tmp_tetrahedron);
            }
        }
        return tetrahedra;
    }

    std::vector<triangle_t> compute_delaunay_triangulation_2D(const std::vector<vector2_t> & points)
    {
        const int ndim = 2;
        std::vector<triangle_t> triangles;
        triangle_t tmp_triangle;
        int *current_index;

        orgQhull::Qhull qhull;
        qhull.runQhull("", ndim, points.size(), (coordT *) points.data(),  "d Qt Qbb Qz");
        for(const auto & facet : qhull.facetList())
        {
            if(!facet.isUpperDelaunay())
            {
                current_index = &tmp_triangle[0];
                for(const auto & vertex : facet.vertices())
                {
                    *current_index++ = vertex.point().id();
                }
                triangles.push_back(tmp_triangle);
            }
        }
        return triangles;
    }

    const std::vector<triangle_t>& Geometry::triangulation(int n_cell_step)
    {
        // Only every n_cell_step'th cell is used. So we check if there is still enough cells in all
        //      directions. Note: when visualising, 'n_cell_step' can be used to e.g. olny visualise
        //      every 2nd spin.
        if ( (n_cells[0]/n_cell_step < 2 && n_cells[0] > 1) ||
             (n_cells[1]/n_cell_step < 2 && n_cells[1] > 1) ||
             (n_cells[2]/n_cell_step < 2 && n_cells[2] > 1) )
        {
            _triangulation.clear();
            return _triangulation;
        }

        // 2D: triangulation
        if (dimensionality == 2)
        {
            // Check if the tetrahedra for this combination of n_cells and n_cell_step has already been calculated
            if (this->last_update_n_cell_step != n_cell_step ||
                this->last_update_n_cells[0]  != n_cells[0]  ||
                this->last_update_n_cells[1]  != n_cells[1]  ||
                this->last_update_n_cells[2]  != n_cells[2]  )
            {
                this->last_update_n_cell_step = n_cell_step;
                this->last_update_n_cells[0]  = n_cells[0];
                this->last_update_n_cells[1]  = n_cells[1];
                this->last_update_n_cells[2]  = n_cells[2];
                
                _triangulation.clear();

                std::vector<vector2_t> points;
                points.resize(spin_pos.size());

                int icell = 0, idx;
                for (int cell_c=0; cell_c<n_cells[2]; cell_c+=n_cell_step)
                {
                    for (int cell_b=0; cell_b<n_cells[1]; cell_b+=n_cell_step)
                    {
                        for (int cell_a=0; cell_a<n_cells[0]; cell_a+=n_cell_step)
                        {
                            for (int ibasis=0; ibasis < n_spins_basic_domain; ++ibasis)
                            {
                                idx = ibasis + n_spins_basic_domain*cell_a + n_spins_basic_domain*n_cells[0]*cell_b + n_spins_basic_domain*n_cells[0]*n_cells[1]*cell_c;
                                points[icell].x = spin_pos[idx][0];
                                points[icell].y = spin_pos[idx][1];
                                ++icell;
                            }
                        }
                    }
                }
                _triangulation = compute_delaunay_triangulation_2D(points);
            }
        }// endif 2D
        // 0D, 1D and 3D give no triangulation
        else
        {
            _triangulation.clear();
        }
        return _triangulation;
    }

    const std::vector<tetrahedron_t>& Geometry::tetrahedra(int n_cell_step)
    {
        // Only every n_cell_step'th cell is used. So we check if there is still enough cells in all
        //      directions. Note: when visualising, 'n_cell_step' can be used to e.g. olny visualise
        //      every 2nd spin.
        if (n_cells[0]/n_cell_step < 2 || n_cells[1]/n_cell_step < 2 || n_cells[2]/n_cell_step < 2)
        {
            _tetrahedra.clear();
            return _tetrahedra;
        }

        // 3D: Tetrahedra
        if (dimensionality == 3)
        {
            // Check if the tetrahedra for this combination of n_cells and n_cell_step has already been calculated
            if (this->last_update_n_cell_step != n_cell_step ||
                this->last_update_n_cells[0]  != n_cells[0]  ||
                this->last_update_n_cells[1]  != n_cells[1]  ||
                this->last_update_n_cells[2]  != n_cells[2]  )
            {
                this->last_update_n_cell_step = n_cell_step;
                this->last_update_n_cells[0]  = n_cells[0];
                this->last_update_n_cells[1]  = n_cells[1];
                this->last_update_n_cells[2]  = n_cells[2];

                // If we have only one spin in the basis our lattice is a simple regular geometry
                bool is_simple_regular_geometry = n_spins_basic_domain == 1;

                // If we have a simple regular geometry everything can be calculated by hand
                if (is_simple_regular_geometry)
                {
                    _tetrahedra.clear();
                    int cell_indices[] = {
                        0, 1, 5, 3,
                        1, 3, 2, 5,
                        3, 2, 5, 6,
                        7, 6, 5, 3,
                        4, 7, 5, 3,
                        0, 4, 3, 5
                        };
                    int x_offset = 1;
                    int y_offset = n_cells[0]/n_cell_step;
                    int z_offset = n_cells[0]/n_cell_step*n_cells[1]/n_cell_step;
                    int offsets[] = {
                        0, x_offset, x_offset+y_offset, y_offset,
                        z_offset, x_offset+z_offset, x_offset+y_offset+z_offset, y_offset+z_offset
                        };
                
                    for (int ix = 0; ix < (n_cells[0]-1)/n_cell_step; ix++)
                    {
                        for (int iy = 0; iy < (n_cells[1]-1)/n_cell_step; iy++)
                        {
                            for (int iz = 0; iz < (n_cells[2]-1)/n_cell_step; iz++)
                            {
                                int base_index = ix*x_offset+iy*y_offset+iz*z_offset;
                                for (int j = 0; j < 6; j++)
                                {
                                    tetrahedron_t tetrahedron;
                                    for (int k = 0; k < 4; k++)
                                    {
                                        int index = base_index + offsets[cell_indices[j*4+k]];
                                        tetrahedron[k] = index;
                                    }
                                    _tetrahedra.push_back(tetrahedron);
                                }
                            }
                        }
                    }
                }
                // For general basis cells we calculate the Delaunay tetrahedra
                else 
                {
                    std::vector<vector3_t> points;
                    points.resize(spin_pos.size());

                    int icell = 0, idx;
                    for (int cell_c=0; cell_c<n_cells[2]; cell_c+=n_cell_step)
                    {
                        for (int cell_b=0; cell_b<n_cells[1]; cell_b+=n_cell_step)
                        {
                            for (int cell_a=0; cell_a<n_cells[0]; cell_a+=n_cell_step)
                            {
                                for (int ibasis=0; ibasis < n_spins_basic_domain; ++ibasis)
                                {
                                    idx = ibasis + n_spins_basic_domain*cell_a + n_spins_basic_domain*n_cells[0]*cell_b + n_spins_basic_domain*n_cells[0]*n_cells[1]*cell_c;
                                    points[icell].x = spin_pos[idx][0];
                                    points[icell].y = spin_pos[idx][1];
                                    points[icell].z = spin_pos[idx][2];
                                    ++icell;
                                }
                            }
                        }
                    }
                    _tetrahedra = compute_delaunay_triangulation_3D(points);
                }
            }
        } // endif 3D
        // 0-2 D gives no tetrahedra
        else
        {
            _tetrahedra.clear();
        }
        return _tetrahedra;
    }

    int Geometry::calculateDimensionality() const
    {
        int dims_basis = 0, dims_translations = 0;
        Vector3 test_vec_basis, test_vec_translations;

        // ----- Find dimensionality of the basis -----
        if (n_spins_basic_domain == 1) dims_basis = 0;
        else if (n_spins_basic_domain == 2) dims_basis = 1;
        else if (n_spins_basic_domain == 3) dims_basis = 2;
        else
        {
            // Get basis atoms relative to the first atom
            Vector3 v0 = basis_atoms[0];
            std::vector<Vector3> b_vectors(n_spins_basic_domain-1);
            for (int i = 1; i < n_spins_basic_domain; ++i)
            {
                b_vectors[i-1] = basis_atoms[i] - v0;
            }
            // Calculate basis dimensionality
            // test vec is along line
            test_vec_basis = b_vectors[0];
            //		is it 1D?
            int n_parallel = 0;
            for (unsigned int i = 1; i < b_vectors.size(); ++i)
            {
                if (std::abs(b_vectors[i].dot(test_vec_basis) - 1.0) < 1e-9) ++n_parallel;
                // else n_parallel will give us the last parallel vector
                // also the if-statement for dims_basis=1 wont be met
                else break;
            }
            if (n_parallel == b_vectors.size() - 1)
            {
                dims_basis = 1;
            }
            else
            {
                // test vec is normal to plane
                test_vec_basis = b_vectors[0].cross(b_vectors[n_parallel+1]);
                //		is it 2D?
                int n_in_plane = 0;
                for (unsigned int i = 2; i < b_vectors.size(); ++i)
                {
                    if (std::abs(b_vectors[i].dot(test_vec_basis)) < 1e-9) ++n_in_plane;
                }
                if (n_in_plane == b_vectors.size() - 2)
                {
                    dims_basis = 2;
                }
                else return 3;
            }
        }


        // ----- Find dimensionality of the translations -----
        //		The following are zero if the corresponding pair is parallel
        double t01, t02, t12;
        t01 = std::abs(translation_vectors[0].dot(translation_vectors[1]) - 1.0);
        t02 = std::abs(translation_vectors[0].dot(translation_vectors[2]) - 1.0);
        t12 = std::abs(translation_vectors[1].dot(translation_vectors[2]) - 1.0);
        //		Check if pairs are linearly independent
        int n_independent_pairs = 0;
        if (t01>1e-9 && n_cells[0] > 1 && n_cells[1] > 1) ++n_independent_pairs;
        if (t02>1e-9 && n_cells[0] > 1 && n_cells[2] > 1) ++n_independent_pairs;
        if (t12>1e-9 && n_cells[1] > 1 && n_cells[2] > 1) ++n_independent_pairs;
        //		Calculate translations dimensionality
        if (n_cells[0] == 1 && n_cells[1] == 1 && n_cells[2] == 1) dims_translations = 0;
        else if (n_independent_pairs == 0)
        {
            dims_translations = 1;
            // test vec is along the line
            for (int i=0; i<3; ++i) if (n_cells[i] > 1) test_vec_translations = translation_vectors[i];
        }
        else if (n_independent_pairs < 3)
        {
            dims_translations = 2;
            // test vec is normal to plane
            int n = 0;
            std::vector<Vector3> plane(2);
            for (int i = 0; i < 3; ++i)
            {
                if (n_cells[i] > 1) plane[n] = translation_vectors[i];
                ++n;
            }
            test_vec_translations = plane[0].cross(plane[1]);
        }
        else return 3;


        // ----- Calculate dimensionality of system -----
        test_vec_basis.normalize();
        test_vec_translations.normalize();
        //		If one dimensionality is zero, only the other counts
        if (dims_basis == 0) return dims_translations;
        else if (dims_translations == 0) return dims_basis;
        //		If both are linear or both are planar, the test vectors should be parallel if the geometry is 1D or 2D
        else if (dims_basis == dims_translations)
        {
            if (std::abs(test_vec_basis.dot(test_vec_translations) - 1.0) < 1e-9) return dims_basis;
            else if (dims_basis == 1) return 2;
            else if (dims_basis == 2) return 3;
        }
        //		If one is linear (1D), and the other planar (2D) then the test vectors should be orthogonal if the geometry is 2D
        else if ( (dims_basis == 1 && dims_translations == 2) || (dims_basis == 2 && dims_translations == 1) )
        {
            if (std::abs(test_vec_basis.dot(test_vec_translations)) < 1e-9) return 2;
            else return 3;
        }

        // We should never get here
        throw Utility::Exception::Unknown_Exception;
        return 0;
    }
}

