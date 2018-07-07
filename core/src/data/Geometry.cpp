#include <data/Geometry.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Qhull.h"
#include "QhullFacetList.h"
#include "QhullVertexSet.h"

#include <array>

namespace Data
{
    Geometry::Geometry(std::vector<Vector3> bravais_vectors, intfield n_cells,
                        std::vector<Vector3> cell_atoms, scalarfield mu_s,
                        intfield cell_atom_types, scalar lattice_constant) :
        bravais_vectors(bravais_vectors), n_cells(n_cells), n_cell_atoms(cell_atoms.size()),
        mu_s(mu_s), cell_atoms(cell_atoms), lattice_constant(lattice_constant),
        nos(cell_atoms.size() * n_cells[0] * n_cells[1] * n_cells[2]), cell_atom_types(cell_atom_types),
        n_cells_total(n_cells[0] * n_cells[1] * n_cells[2])
    {
        for (int iatom = 0; iatom < n_cell_atoms; ++iatom)
        {
            // Get x,y,z of component of atom positions in unit of length (instead of in units of a,b,c)
            Vector3 build_array = bravais_vectors[0] * cell_atoms[iatom][0] + bravais_vectors[1] * cell_atoms[iatom][1] + bravais_vectors[2] * cell_atoms[iatom][2];
            cell_atoms[iatom] = lattice_constant * build_array;
        }

        // Generate positions and atom types
        this->positions = vectorfield(nos);
        this->atom_types = intfield(nos, 0);
        Engine::Vectormath::Build_Spins(positions, atom_types, cell_atoms, cell_atom_types, bravais_vectors, n_cells);

        // Calculate some info
        this->calculateBounds();
        this->calculateUnitCellBounds();
        this->calculateDimensionality();

        // Calculate center of the System
        this->center = 0.5 *  (this->bounds_min + this->bounds_max);

        // Calculate the type of geometry
        this->calculateGeometryType();

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
        if (this->dimensionality == 2)
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
                points.resize(positions.size());

                int icell = 0, idx;
                for (int cell_c=0; cell_c<n_cells[2]; cell_c+=n_cell_step)
                {
                    for (int cell_b=0; cell_b<n_cells[1]; cell_b+=n_cell_step)
                    {
                        for (int cell_a=0; cell_a<n_cells[0]; cell_a+=n_cell_step)
                        {
                            for (int ibasis=0; ibasis < n_cell_atoms; ++ibasis)
                            {
                                idx = ibasis + n_cell_atoms*cell_a + n_cell_atoms*n_cells[0]*cell_b + n_cell_atoms*n_cells[0]*n_cells[1]*cell_c;
                                points[icell].x = positions[idx][0];
                                points[icell].y = positions[idx][1];
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
        if (this->dimensionality == 3)
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
                bool is_simple_regular_geometry = n_cell_atoms == 1;

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
                    points.resize(positions.size());

                    int icell = 0, idx;
                    for (int cell_c=0; cell_c<n_cells[2]; cell_c+=n_cell_step)
                    {
                        for (int cell_b=0; cell_b<n_cells[1]; cell_b+=n_cell_step)
                        {
                            for (int cell_a=0; cell_a<n_cells[0]; cell_a+=n_cell_step)
                            {
                                for (int ibasis=0; ibasis < n_cell_atoms; ++ibasis)
                                {
                                    idx = ibasis + n_cell_atoms*cell_a + n_cell_atoms*n_cells[0]*cell_b + n_cell_atoms*n_cells[0]*n_cells[1]*cell_c;
                                    points[icell].x = positions[idx][0];
                                    points[icell].y = positions[idx][1];
                                    points[icell].z = positions[idx][2];
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


    std::vector<Vector3> Geometry::BravaisVectorsSC()
    {
        return { { 1,0,0 },
                 { 0,1,0 },
                 { 0,0,1 } };
    }

    std::vector<Vector3> Geometry::BravaisVectorsFCC()
    {
        return { { 0.5,0.0,0.5 },
                 { 0.5,0.5,0.0 },
                 { 0.0,0.5,0.5 } };
    }

    std::vector<Vector3> Geometry::BravaisVectorsBCC()
    {
        return { { 0.5, 0.5,-0.5 },
                 { -0.5, 0.5,-0.5 },
                 { 0.5,-0.5, 0.5 } };
    }

    std::vector<Vector3> Geometry::BravaisVectorsHex2D60()
    {
        return { { 0.5*std::sqrt(3), -0.5, 0 },
                 { 0.5*std::sqrt(3),  0.5, 0 },
                 { 0,   0,                 1 } };
    }

    std::vector<Vector3> Geometry::BravaisVectorsHex2D120()
    {
        return { { 0.5, -0.5*std::sqrt(3), 0 },
                 { 0.5,  0.5*std::sqrt(3), 0 },
                 { 0,    0,                1 } };
    }


    void Geometry::calculateDimensionality()
    {
        int dims_basis = 0, dims_translations = 0;
        Vector3 test_vec_basis, test_vec_translations;

        // ----- Find dimensionality of the basis -----
        if      (n_cell_atoms == 1) dims_basis = 0;
        else if (n_cell_atoms == 2) dims_basis = 1;
        else if (n_cell_atoms == 3) dims_basis = 2;
        else
        {
            // Get basis atoms relative to the first atom
            Vector3 v0 = cell_atoms[0];
            std::vector<Vector3> b_vectors(n_cell_atoms-1);
            for (int i = 1; i < n_cell_atoms; ++i)
            {
                b_vectors[i-1] = cell_atoms[i] - v0;
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
                else
                {
                    this->dimensionality = 3;
                    return;
                }
            }
        }


        // ----- Find dimensionality of the translations -----
        //		The following are zero if the corresponding pair is parallel
        double t01, t02, t12;
        t01 = std::abs(bravais_vectors[0].dot(bravais_vectors[1]) - 1.0);
        t02 = std::abs(bravais_vectors[0].dot(bravais_vectors[2]) - 1.0);
        t12 = std::abs(bravais_vectors[1].dot(bravais_vectors[2]) - 1.0);
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
            for (int i=0; i<3; ++i) if (n_cells[i] > 1) test_vec_translations = bravais_vectors[i];
        }
        else if (n_independent_pairs < 3)
        {
            dims_translations = 2;
            // test vec is normal to plane
            int n = 0;
            std::vector<Vector3> plane(2);
            for (int i = 0; i < 3; ++i)
            {
                if (n_cells[i] > 1) plane[n] = bravais_vectors[i];
                ++n;
            }
            test_vec_translations = plane[0].cross(plane[1]);
        }
        else
        {
            this->dimensionality = 3;
            return;
        }


        // ----- Calculate dimensionality of system -----
        test_vec_basis.normalize();
        test_vec_translations.normalize();
        //		If one dimensionality is zero, only the other counts
        if (dims_basis == 0)
        {
            this->dimensionality = dims_translations;
            return;
        }
        else if (dims_translations == 0)
        {
            this->dimensionality = dims_basis;
            return;
        }
        //		If both are linear or both are planar, the test vectors should be parallel if the geometry is 1D or 2D
        else if (dims_basis == dims_translations)
        {
            if (std::abs(test_vec_basis.dot(test_vec_translations) - 1.0) < 1e-9)
            {
                this->dimensionality = dims_basis;
                return;
            }
            else if (dims_basis == 1)
            {
                this->dimensionality = 2;
                return;
            }
            else if (dims_basis == 2)
            {
                this->dimensionality = 3;
                return;
            }
        }
        //		If one is linear (1D), and the other planar (2D) then the test vectors should be orthogonal if the geometry is 2D
        else if ( (dims_basis == 1 && dims_translations == 2) || (dims_basis == 2 && dims_translations == 1) )
        {
            if (std::abs(test_vec_basis.dot(test_vec_translations)) < 1e-9)
            {
                this->dimensionality = 2;
                return;
            }
            else
            {
                this->dimensionality = 3;
                return;
            }
        }
    }

    void Geometry::calculateBounds()
    {
        this->bounds_max.setZero();
        this->bounds_min.setZero();
        for (int iatom = 0; iatom < nos; ++iatom)
        {
            for (int dim = 0; dim < 3; ++dim)
            {
                if (this->positions[iatom][dim] < this->bounds_min[dim]) this->bounds_min[dim] = positions[iatom][dim];
                if (this->positions[iatom][dim] > this->bounds_max[dim]) this->bounds_max[dim] = positions[iatom][dim];
            }
        }
    }

    void Geometry::calculateUnitCellBounds()
    {
        this->cell_bounds_max.setZero();
        this->cell_bounds_min.setZero();
        for (unsigned int ivec = 0; ivec < bravais_vectors.size(); ++ivec)
        {
            for (int iatom = 0; iatom < n_cell_atoms; ++iatom)
            {
                auto neighbour1 = cell_atoms[iatom] + bravais_vectors[ivec];
                auto neighbour2 = cell_atoms[iatom] - bravais_vectors[ivec];
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
    }

    void Geometry::calculateGeometryType()
    {
        // Automatically try to determine GeometryType
        // Single-atom unit cell
        if (cell_atoms.size() == 1)
        {
            // If the basis vectors are orthogonal, it is a rectilinear lattice
            if (std::abs(bravais_vectors[0].dot(bravais_vectors[1])) < 1e-6 &&
                std::abs(bravais_vectors[0].dot(bravais_vectors[2])) < 1e-6)
            {
                // If equidistant it is simple cubic
                if (bravais_vectors[0].norm() == bravais_vectors[1].norm() == bravais_vectors[2].norm())
                    this->classifier = BravaisLatticeType::SC;
                // Otherwise only rectilinear
                else
                    this->classifier = BravaisLatticeType::Rectilinear;
            }
        }
        // Regular unit cell with multiple atoms (e.g. bcc, fcc, hex)
        //else if (n_cell_atoms == 2)
        // Irregular unit cells arranged on a lattice (e.g. B20 or custom)
        /*else if (n_cells[0] > 1 || n_cells[1] > 1 || n_cells[2] > 1)
        {
            this->classifier = BravaisLatticeType::Lattice;
        }*/
        // A single irregular unit cell
        else
        {
            this->classifier = BravaisLatticeType::Irregular;
        }
    }
}

