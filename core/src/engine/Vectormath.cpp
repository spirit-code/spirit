#ifndef SPIRIT_USE_CUDA

#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <algorithm>

using namespace Utility;
using Utility::Constants::Pi;

namespace Engine
{
    namespace Vectormath
    {
        scalar angle(const Vector3 & v1, const Vector3 & v2)
        {
            scalar r = v1.dot(v2);
            // Prevent NaNs from occurring
            r = std::fmax(-1.0, std::fmin(1.0, r));
            // Angle
            return std::acos(r);
        }

        void rotate(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out)
        {
            v_out = v * std::cos(angle) + axis.cross(v) * std::sin(angle) +
                    axis * axis.dot(v) * (1 - std::cos(angle));
        }

        // XXX: should we add test for that function since it's calling the already tested rotat()
        void rotate( const vectorfield & v, const vectorfield & axis, const scalarfield & angle,
                     vectorfield & v_out )
        {
            for( unsigned int i=0; i<v_out.size(); i++)
                rotate( v[i], axis[i], angle[i], v_out[i] );
        }

        Vector3 decompose(const Vector3 & v, const std::vector<Vector3> & basis)
        {
            Eigen::Ref<const Matrix3> A = Eigen::Map<const Matrix3>(basis[0].data());
            return A.colPivHouseholderQr().solve(v);
        }

        /////////////////////////////////////////////////////////////////


        std::array<scalar,3> Magnetization(const vectorfield & vf)
        {
            Vector3 vfmean = mean(vf);
            std::array<scalar, 3> M{vfmean[0], vfmean[1], vfmean[2]};
            return M;
        }

        scalar solid_angle_1(const Vector3 & v1, const Vector3 & v2, const Vector3 & v3)
        {
            // Get sign
            scalar pm = v1.dot(v2.cross(v3));
            if (pm != 0) pm /= std::abs(pm);

            // angle
            scalar solid_angle = ( 1 + v1.dot(v2) + v2.dot(v3) + v3.dot(v1) ) /
                                std::sqrt( 2 * (1+v1.dot(v2)) * (1+v2.dot(v3)) * (1+v3.dot(v1)) );
            if (solid_angle == 1)
                solid_angle = 0;
            else
                solid_angle = pm * 2 * std::acos(solid_angle);

            return solid_angle;
        }

        scalar solid_angle_2(const Vector3 & v1, const Vector3 & v2, const Vector3 & v3)
        {
            // Using the solid angle formula by Oosterom and Strackee (note we assume vectors to be normalized to 1)
            // https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron

            scalar x = v1.dot(v2.cross(v3));
            scalar y = 1 + v1.dot(v2) + v1.dot(v3) + v2.dot(v3);
            scalar solid_angle = 2 * std::atan2( x , y );

            return solid_angle;
        }

        scalar TopologicalCharge(const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions)
        {
            // This implementations assumes
            // 1. No basis atom lies outside the cell spanned by the basis vectors of the lattice
            // 2. The geometry is a plane in x and y and spanned by the first 2 basis_vectors of the lattice
            // 3. The first basis atom lies at (0,0)
            
            const auto & positions = geometry.positions;
            scalar charge = 0;

            // Compute Delaunay for unitcell + basis with neighbouring lattice sites in directions a, b, and a+b
            std::vector<Data::vector2_t> basis_cell_points(geometry.n_cell_atoms + 3);
            for(int i = 0; i < geometry.n_cell_atoms; i++)
            {
                basis_cell_points[i].x = double(positions[i][0]);
                basis_cell_points[i].y = double(positions[i][1]);
            }

            // To avoid cases where the basis atoms lie on the boundary of the convex hull the corners of the parallelogram
            // spanned by the lattice sites 0, a, b and a+b are stretched away from the center for the triangulation
            scalar stretch_factor = 0.1;

            // For the rare case where the first basis atoms does not lie at (0,0,0)
            Vector3 basis_offset = positions[0];

            Vector3 ta = geometry.lattice_constant * geometry.bravais_vectors[0];
            Vector3 tb = geometry.lattice_constant * geometry.bravais_vectors[1];
            Vector3 tc = geometry.lattice_constant * geometry.bravais_vectors[2];

            // basis_cell_points[0] coincides with the '0' lattice site (plus basis_offset)
            basis_cell_points[0].x -= stretch_factor * (ta + tb)[0];
            basis_cell_points[0].y -= stretch_factor * (ta + tb)[1];

            // a+b
            basis_cell_points[geometry.n_cell_atoms].x   = double((ta + tb + positions[0] + stretch_factor * (ta + tb))[0]);
            basis_cell_points[geometry.n_cell_atoms].y   = double((ta + tb + positions[0] + stretch_factor * (ta + tb))[1]);
            // b
            basis_cell_points[geometry.n_cell_atoms+1].x = double((tb + positions[0] - stretch_factor * (ta - tb))[0]);
            basis_cell_points[geometry.n_cell_atoms+1].y = double((tb + positions[0] - stretch_factor * (ta - tb))[1]);
            // a
            basis_cell_points[geometry.n_cell_atoms+2].x = double((ta + positions[0] + stretch_factor * (ta - tb))[0]);
            basis_cell_points[geometry.n_cell_atoms+2].y = double((ta + positions[0] + stretch_factor * (ta - tb))[1]);

            std::vector<Data::triangle_t> triangulation;
            triangulation = Data::compute_delaunay_triangulation_2D(basis_cell_points);

            for(Data::triangle_t tri : triangulation)
            {
                // Compute the sign of this triangle
                Vector3 triangle_normal;
                vectorfield tri_positions(3);
                for(int i=0; i<3; i++)
                {
                    tri_positions[i] = {basis_cell_points[tri[i]].x, basis_cell_points[tri[i]].y, 0};
                }
                triangle_normal = (tri_positions[0]-tri_positions[1]).cross(tri_positions[0] - tri_positions[2]);
                triangle_normal.normalize();
                scalar sign = triangle_normal[2]/std::abs(triangle_normal[2]);

                // We try to apply the Delaunay triangulation at each bravais-lattice point
                // For each corner of the triangle we check wether it is "allowed" (which means either inside the simulation box or permitted by periodic boundary conditions)
                // Then we can add the top charge for all trios of spins connected by this triangle
                for(int b = 0; b < geometry.n_cells[1]; ++b)
                {
                    for(int a = 0; a < geometry.n_cells[0]; ++a)
                    {
                        std::array<Vector3, 3> tri_spins;
                        // bools to check wether it is allowed to take the next lattice site in direction a, b or a+b
                        bool a_next_allowed = (a+1 < geometry.n_cells[0] || boundary_conditions[0]);
                        bool b_next_allowed = (b+1 < geometry.n_cells[1] || boundary_conditions[1]);
                        bool valid_triangle = true;
                        for(int i = 0; i<3; ++i)
                        {
                            int idx;
                            if(tri[i] < geometry.n_cell_atoms) // tri[i] is an index of a basis atom, no wrap around can occur
                            {
                                idx = (tri[i] + a * geometry.n_cell_atoms + b * geometry.n_cell_atoms * geometry.n_cells[0]);
                            }
                            else if (tri[i] == geometry.n_cell_atoms + 2 && a_next_allowed) // Translation by a
                            {
                                idx = ((a + 1) % geometry.n_cells[0]) * geometry.n_cell_atoms + b * geometry.n_cell_atoms * geometry.n_cells[0];
                            }
                            else if (tri[i] == geometry.n_cell_atoms + 1 && b_next_allowed) // Translation by b
                            {
                                idx = a * geometry.n_cell_atoms + ((b + 1) % geometry.n_cells[1]) * geometry.n_cell_atoms * geometry.n_cells[0];
                            }
                            else if (tri[i] == geometry.n_cell_atoms && a_next_allowed && b_next_allowed) // Translation by a + b
                            {
                                idx = ((a + 1) % geometry.n_cells[0]) * geometry.n_cell_atoms + ((b + 1) % geometry.n_cells[1]) * geometry.n_cell_atoms * geometry.n_cells[0];
                            }
                            else // Translation not allowed, skip to next triangle
                            {
                                valid_triangle = false;
                                break;
                            }
                            tri_spins[i] = vf[idx];
                        }
                        if(valid_triangle)
                            charge += sign * solid_angle_2(tri_spins[0], tri_spins[1], tri_spins[2]);
                    }
                }
            }
            return charge / (4*Pi);
        }

        // Utility function for the SIB Solver
        void transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
        {
            #pragma omp parallel for
            for (unsigned int i = 0; i < spins.size(); ++i)
            {
                Vector3 A = 0.5 * force[i];

                // 1/determinant(A)
                scalar detAi = 1.0 / (1 + A.squaredNorm());

                // calculate equation without the predictor?
                Vector3 a2 = spins[i] - spins[i].cross(A);

                out[i][0] = (a2[0] * (A[0] * A[0] + 1   ) + a2[1] * (A[0] * A[1] - A[2]) + a2[2] * (A[0] * A[2] + A[1])) * detAi;
                out[i][1] = (a2[0] * (A[1] * A[0] + A[2]) + a2[1] * (A[1] * A[1] + 1   ) + a2[2] * (A[1] * A[2] - A[0])) * detAi;
                out[i][2] = (a2[0] * (A[2] * A[0] - A[1]) + a2[1] * (A[2] * A[1] + A[0]) + a2[2] * (A[2] * A[2] + 1   )) * detAi;
            }
        }

        void get_random_vector(std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec)
        {
            for (int dim = 0; dim < 3; ++dim)
            {
                vec[dim] = distribution(prng);
            }
        }
        void get_random_vectorfield(std::mt19937 & prng, vectorfield & xi)
        {
            // PRNG gives RN [-1,1] -> multiply with epsilon
            auto distribution = std::uniform_real_distribution<scalar>(-1, 1);
            // TODO: parallelization of this is actually not quite so trivial
            #pragma omp parallel for
            for (unsigned int i = 0; i < xi.size(); ++i)
            {
                get_random_vector(distribution, prng, xi[i]);
            }
        }

        void get_random_vector_unitsphere(std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec)
        {
            scalar v_z = distribution(prng);
            scalar phi = distribution(prng)*Pi;

            scalar r_xy = std::sqrt(1 - v_z*v_z);

            vec[0] = r_xy * std::cos(phi);
            vec[1] = r_xy * std::sin(phi);
            vec[2] = v_z;
        }
        void get_random_vectorfield_unitsphere(std::mt19937 & prng, vectorfield & xi)
        {
            // PRNG gives RN [-1,1] -> multiply with epsilon
            auto distribution = std::uniform_real_distribution<scalar>(-1, 1);
            // TODO: parallelization of this is actually not quite so trivial
            #pragma omp parallel for
            for (unsigned int i = 0; i < xi.size(); ++i)
            {
                get_random_vector_unitsphere(distribution, prng, xi[i]);
            }
        }

        void get_gradient_distribution(const Data::Geometry & geometry, Vector3 gradient_direction, scalar gradient_start, scalar gradient_inclination, scalarfield & distribution, scalar range_min, scalar range_max)
        {
            // Ensure a normalized direction vector
            gradient_direction.normalize();

            // Basic linear gradient distribution
            set_c_dot(gradient_inclination, gradient_direction, geometry.positions, distribution);

            // Get the minimum (i.e. starting point) of the distribution
            scalar bmin = geometry.bounds_min.dot(gradient_direction);
            scalar bmax = geometry.bounds_max.dot(gradient_direction);
            scalar dist_min = std::min(bmin, bmax);
            // Set the starting point
            add(distribution, gradient_start - gradient_inclination*dist_min);

            // Cut off negative values
            set_range(distribution, range_min, range_max);
        }


        void directional_gradient(const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions, const Vector3 & direction, vectorfield & gradient)
        {
            // std::cout << "start gradient" << std::endl;
            vectorfield translations = { { 0,0,0 }, { 0,0,0 }, { 0,0,0 } };
            auto& n_cells = geometry.n_cells;

            neighbourfield neigh;

            // TODO: calculate Neighbours outside iterations
            // Neighbours::get_Neighbours(geometry, neigh);

            // TODO: proper usage of neighbours
            // Hardcoded neighbours - for spin current in a rectangular lattice
            neigh = neighbourfield(0);
            Neighbour neigh_tmp;
            neigh_tmp.i = 0;
            neigh_tmp.j = 0;
            neigh_tmp.idx_shell = 0;

            neigh_tmp.translations[0] = 1;
            neigh_tmp.translations[1] = 0;
            neigh_tmp.translations[2] = 0;
            neigh.push_back(neigh_tmp);

            neigh_tmp.translations[0] = -1;
            neigh_tmp.translations[1] = 0;
            neigh_tmp.translations[2] = 0;
            neigh.push_back(neigh_tmp);

            neigh_tmp.translations[0] = 0;
            neigh_tmp.translations[1] = 1;
            neigh_tmp.translations[2] = 0;
            neigh.push_back(neigh_tmp);

            neigh_tmp.translations[0] = 0;
            neigh_tmp.translations[1] = -1;
            neigh_tmp.translations[2] = 0;
            neigh.push_back(neigh_tmp);

            neigh_tmp.translations[0] = 0;
            neigh_tmp.translations[1] = 0;
            neigh_tmp.translations[2] = 1;
            neigh.push_back(neigh_tmp);

            neigh_tmp.translations[0] = 0;
            neigh_tmp.translations[1] = 0;
            neigh_tmp.translations[2] = -1;
            neigh.push_back(neigh_tmp);

            // Loop over vectorfield
            for(unsigned int ispin = 0; ispin < vf.size(); ++ispin)
            {
                auto translations_i = translations_from_idx(n_cells, geometry.n_cell_atoms, ispin); // transVec of spin i
                // int k = i%geometry.n_cell_atoms; // index within unit cell - k=0 for all cases used in the thesis
                scalar n = 0;

                gradient[ispin].setZero();

                std::vector<Vector3> euclidean { {1,0,0}, {0,1,0}, {0,0,1} };
                std::vector<Vector3> contrib = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
                Vector3 proj = {0, 0, 0};
                Vector3 projection_inv = {0, 0, 0};

                // TODO: both loops together.

                // Loop over neighbours of this vector to calculate contributions of finite differences to current direction
                for( unsigned int j = 0; j < neigh.size(); ++j )
                {
                    if( boundary_conditions_fulfilled(geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations) )
                    {
                        // Index of neighbour
                        int ineigh = idx_from_translations(n_cells, geometry.n_cell_atoms, translations_i, neigh[j].translations);
                        if( ineigh >= 0 )
                        {
                            auto d = geometry.positions[ineigh] - geometry.positions[ispin];
                            for( int dim=0; dim<3; ++dim )
                            {
                                proj[dim] += std::abs(euclidean[dim].dot(d.normalized()));
                            }
                        }
                    }
                }
                for( int dim=0; dim<3; ++dim )
                {
                    if( std::abs(proj[dim]) > 1e-10 )
                        projection_inv[dim] = 1.0/proj[dim];
                }
                // Loop over neighbours of this vector to calculate finite differences
                for( unsigned int j = 0; j < neigh.size(); ++j )
                {
                    if( boundary_conditions_fulfilled(geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations) )
                    {
                        // Index of neighbour
                        int ineigh = idx_from_translations(n_cells, geometry.n_cell_atoms, translations_i, neigh[j].translations);
                        if( ineigh >= 0 )
                        {
                            auto d = geometry.positions[ineigh] - geometry.positions[ispin];
                            for( int dim=0; dim<3; ++dim )
                            {
                                contrib[dim] += euclidean[dim].dot(d) / d.dot(d) * ( vf[ineigh] - vf[ispin] );
                            }
                        }
                    }
                }

                for( int dim=0; dim<3; ++dim )
                {
                    gradient[ispin] += direction[dim]*projection_inv[dim] * contrib[dim];
                }
            }
        }

        /////////////////////////////////////////////////////////////////

        void fill(scalarfield & sf, scalar s)
        {
            #pragma omp parallel for
            for (unsigned int i = 0; i<sf.size(); ++i)
                sf[i] = s;
        }
        void fill(scalarfield & sf, scalar s, const intfield & mask)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<sf.size(); ++i)
                sf[i] = mask[i]*s;
        }

        void scale(scalarfield & sf, scalar s)
        {
            #pragma omp parallel for
            for (unsigned int i = 0; i<sf.size(); ++i)
                sf[i] *= s;
        }

        void add(scalarfield & sf, scalar s)
        {
            #pragma omp parallel for
            for (unsigned int i = 0; i<sf.size(); ++i)
                sf[i] += s;
        }

        scalar sum(const scalarfield & sf)
        {
            scalar ret = 0;
            #pragma omp parallel for reduction(+:ret)
            for (unsigned int i = 0; i<sf.size(); ++i)
                ret += sf[i];
            return ret;
        }

        scalar mean(const scalarfield & sf)
        {
            scalar ret = sum(sf)/sf.size();
            return ret;
        }

        void set_range(scalarfield & sf, scalar sf_min, scalar sf_max)
        {
            #pragma omp parallel for
            for (unsigned int i = 0; i<sf.size(); ++i)
                sf[i] = std::min( std::max( sf_min, sf[i] ), sf_max );
        }

        void fill(vectorfield & vf, const Vector3 & v)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<vf.size(); ++i)
                vf[i] = v;
        }
        void fill(vectorfield & vf, const Vector3 & v, const intfield & mask)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<vf.size(); ++i)
                vf[i] = mask[i]*v;
        }

        void normalize_vectors(vectorfield & vf)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<vf.size(); ++i)
                vf[i].normalize();
        }

        void norm( const vectorfield & vf, scalarfield & norm )
        {
            for (unsigned int i=0; i<vf.size(); ++i)
                norm[i] = vf[i].norm();
        }

        std::pair<scalar, scalar> minmax_component(const vectorfield & v1)
        {
            scalar minval=1e6, maxval=-1e6;
            std::pair<scalar, scalar> minmax;
            #pragma omp parallel for reduction(min: minval) reduction(max : maxval)
            for (unsigned int i = 0; i < v1.size(); ++i)
            {
                for (int dim = 0; dim < 3; ++dim)
                {
                    if (v1[i][dim] < minval) minval = v1[i][dim];
                    if (v1[i][dim] > maxval) maxval = v1[i][dim];
                }
            }
            minmax.first = minval;
            minmax.second = maxval;
            return minmax;
        }
        scalar  max_abs_component(const vectorfield & vf)
        {
            // We want the Maximum of Absolute Values of all force components on all images
            scalar absmax = 0;
            // Find minimum and maximum values
            std::pair<scalar,scalar> minmax = minmax_component(vf);
            // Mamimum of absolute values
            absmax = std::max(absmax, std::abs(minmax.first));
            absmax = std::max(absmax, std::abs(minmax.second));
            // Return
            return absmax;
        }

        void scale(vectorfield & vf, const scalar & sc)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<vf.size(); ++i)
                vf[i] *= sc;
        }

        void scale(vectorfield & vf, const scalarfield & sf, bool inverse)
        {
            if( inverse )
            {
                #pragma omp parallel for
                for (unsigned int i=0; i<vf.size(); ++i)
                    vf[i] /= sf[i];
            }
            else
            {
                #pragma omp parallel for
                for (unsigned int i=0; i<vf.size(); ++i)
                    vf[i] *= sf[i];
            }
        }

        Vector3 sum(const vectorfield & vf)
        {
            Vector3 ret = { 0,0,0 };
            #pragma omp parallel for reduction(+:ret)
            for (unsigned int i = 0; i<vf.size(); ++i)
                ret += vf[i];
            return ret;
        }

        Vector3 mean(const vectorfield & vf)
        {
            Vector3 ret = sum(vf)/vf.size();
            return ret;
        }

        void divide( const scalarfield & numerator, const scalarfield & denominator, scalarfield & out )
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<out.size(); ++i)
                out[i] = numerator[i] / denominator[i];
        }

        // computes the inner product of two vectorfields v1 and v2
        scalar dot(const vectorfield & v1, const vectorfield & v2)
        {
            scalar ret = 0;
            #pragma omp parallel for reduction(+:ret)
            for (unsigned int i = 0; i<v1.size(); ++i)
                ret += v1[i].dot(v2[i]);
            return ret;
        }

        // computes the inner products of vectors in vf1 and vf2
        // vf1 and vf2 are vectorfields
        void dot(const vectorfield & vf1, const vectorfield & vf2, scalarfield & out)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<vf1.size(); ++i)
                out[i] = vf1[i].dot(vf2[i]);
        }

        // computes the product of scalars in s1 and s2
        // s1 and s2 are scalarfields
        void dot( const scalarfield & s1, const scalarfield & s2, scalarfield & out )
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<s1.size(); i++)
                out[i] = s1[i] * s2[i];
        }

        // computes the vector (cross) products of vectors in v1 and v2
        // v1 and v2 are vector fields
        void cross(const vectorfield & v1, const vectorfield & v2, vectorfield & out)
        {
            #pragma omp parallel for
            for (unsigned int i=0; i<v1.size(); ++i)
                out[i] = v1[i].cross(v2[i]);
        }


        // out[i] += c*a
        void add_c_a(const scalar & c, const Vector3 & vec, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c*vec;
        }
        // out[i] += c*a[i]
        void add_c_a(const scalar & c, const vectorfield & vf, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c*vf[idx];
        }
        void add_c_a(const scalar & c, const vectorfield & vf, vectorfield & out, const intfield & mask)
        {
            #pragma omp parallel for
            for (unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += mask[idx] * c*vf[idx];
        }
        // out[i] += c[i]*a[i]
        void add_c_a( const scalarfield & c, const vectorfield & vf, vectorfield & out )
        {
            #pragma omp parallel for
            for( unsigned int idx = 0; idx < out.size(); ++idx )
                out[idx] += c[idx] * vf[idx];
        }

        // out[i] = c*a
        void set_c_a(const scalar & c, const Vector3 & vec, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = c*vec;
        }
        // out[i] = c*a
        void set_c_a(const scalar & c, const Vector3 & vec, vectorfield & out, const intfield & mask)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = mask[idx]*c*vec;
        }

        // out[i] = c*a[i]
        void set_c_a(const scalar & c, const vectorfield & vf, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = c*vf[idx];
        }
        // out[i] = c*a[i]
        void set_c_a(const scalar & c, const vectorfield & vf, vectorfield & out, const intfield & mask)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = mask[idx] * c*vf[idx];
        }
        // out[i] = c[i]*a[i]
        void set_c_a( const scalarfield & c, const vectorfield & vf, vectorfield & out )
        {
            #pragma omp parallel for
            for( unsigned int idx=0; idx < out.size(); ++idx)
                out[idx] = c[idx] * vf[idx];
        }

        // out[i] += c * a*b[i]
        void add_c_dot(const scalar & c, const Vector3 & vec, const vectorfield & vf, scalarfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c*vec.dot(vf[idx]);
        }
        // out[i] += c * a[i]*b[i]
        void add_c_dot(const scalar & c, const vectorfield & vf1, const vectorfield & vf2, scalarfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c*vf1[idx].dot(vf2[idx]);
        }

        // out[i] = c * a*b[i]
        void set_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = c*a.dot(b[idx]);
        }
        // out[i] = c * a[i]*b[i]
        void set_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = c*a[idx].dot(b[idx]);
        }


        // out[i] += c * a x b[i]
        void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c*a.cross(b[idx]);
        }
        // out[i] += c * a[i] x b[i]
        void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c*a[idx].cross(b[idx]);
        }
        // out[i] += c[i] * a[i] x b[i]
        void add_c_cross(const scalarfield & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
            #pragma omp parallel for
            for (unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] += c[idx] * a[idx].cross(b[idx]);
        }

        // out[i] = c * a x b[i]
        void set_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = c*a.cross(b[idx]);
        }
        // out[i] = c * a[i] x b[i]
        void set_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < out.size(); ++idx)
                out[idx] = c*a[idx].cross(b[idx]);
        }
    }
}

#endif