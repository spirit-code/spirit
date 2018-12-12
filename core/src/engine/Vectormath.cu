#ifdef SPIRIT_USE_CUDA

#define EIGEN_USE_GPU

#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>
#include <algorithm>

#include <curand.h>
#include <curand_kernel.h>

using namespace Utility;
using Utility::Constants::Pi;

// CUDA Version
namespace Engine
{
    namespace Vectormath
    {
        /////////////////////////////////////////////////////////////////
        // BOILERPLATE CUDA Reductions

        __inline__ __device__
        scalar warpReduceSum(scalar val)
        {
            for (int offset = warpSize/2; offset > 0; offset /= 2)
                val += __shfl_down(val, offset);
            return val;
        }

        __inline__ __device__
        scalar blockReduceSum(scalar val)
        {
            static __shared__ scalar shared[32]; // Shared mem for 32 partial sums
            int lane = threadIdx.x % warpSize;
            int wid = threadIdx.x / warpSize;

            val = warpReduceSum(val);     // Each warp performs partial reduction

            if (lane==0) shared[wid]=val; // Write reduced value to shared memory

            __syncthreads();              // Wait for all partial reductions

            //read from shared memory only if that warp existed
            val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

            if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

            return val;
        }

        __global__ void cu_sum(const scalar *in, scalar* out, int N)
        {
            scalar sum = int(0);
            for(int i = blockIdx.x * blockDim.x + threadIdx.x;
                i < N;
                i += blockDim.x * gridDim.x)
            {
                sum += in[i];
            }
            sum = blockReduceSum(sum);
            if (threadIdx.x == 0)
                atomicAdd(out, sum);
        }



        __inline__ __device__
        Vector3 warpReduceSum(Vector3 val)
        {
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                val[0] += __shfl_down(val[0], offset);
                val[1] += __shfl_down(val[1], offset);
                val[2] += __shfl_down(val[2], offset);
            }
            return val;
        }

        __inline__ __device__
        Vector3 blockReduceSum(Vector3 val)
        {
            static __shared__ Vector3 shared[32]; // Shared mem for 32 partial sums
            int lane = threadIdx.x % warpSize;
            int wid = threadIdx.x / warpSize;

            val = warpReduceSum(val);     // Each warp performs partial reduction

            if (lane==0) shared[wid]=val; // Write reduced value to shared memory

            __syncthreads();              // Wait for all partial reductions

            // Read from shared memory only if that warp existed
            val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : Vector3{0,0,0};

            if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

            return val;
        }

        __global__ void cu_sum(const Vector3 *in, Vector3* out, int N)
        {
            Vector3 sum{0,0,0};
            for(int i = blockIdx.x * blockDim.x + threadIdx.x;
                i < N;
                i += blockDim.x * gridDim.x)
            {
                sum += in[i];
            }
            sum = blockReduceSum(sum);
            if (threadIdx.x == 0)
            {
                atomicAdd(&out[0][0], sum[0]);
                atomicAdd(&out[0][1], sum[1]);
                atomicAdd(&out[0][2], sum[2]);
            }
        }


        __inline__ __device__
        scalar warpReduceMin(scalar val)
        {
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                val  = min(val, __shfl_down(val, offset));
            }
            return val;
        }
        __inline__ __device__
        scalar warpReduceMax(scalar val)
        {
            for (int offset = warpSize/2; offset > 0; offset /= 2)
            {
                val = max(val, __shfl_down(val, offset));
            }
            return val;
        }

        __inline__ __device__
        void blockReduceMinMax(scalar val, scalar *out_min, scalar *out_max)
        {
            static __shared__ scalar shared_min[32]; // Shared mem for 32 partial minmax comparisons
            static __shared__ scalar shared_max[32]; // Shared mem for 32 partial minmax comparisons

            int lane = threadIdx.x % warpSize;
            int wid = threadIdx.x / warpSize;

            scalar _min = warpReduceMin(val);  // Each warp performs partial reduction
            scalar _max = warpReduceMax(val);  // Each warp performs partial reduction

            if (lane==0) shared_min[wid]=_min;  // Write reduced minmax to shared memory
            if (lane==0) shared_max[wid]=_max;  // Write reduced minmax to shared memory
            __syncthreads();                      // Wait for all partial reductions

            // Read from shared memory only if that warp existed
            _min  = (threadIdx.x < blockDim.x / warpSize) ? shared_min[lane] : 0;
            _max  = (threadIdx.x < blockDim.x / warpSize) ? shared_max[lane] : 0;

            if (wid==0) _min  = warpReduceMin(_min);  // Final minmax reduce within first warp
            if (wid==0) _max  = warpReduceMax(_max);  // Final minmax reduce within first warp

            out_min[0] = _min;
            out_max[0] = _max;
        }

        __global__ void cu_MinMax(const scalar *in, scalar* out_min, scalar* out_max, int N)
        {
            scalar tmp, tmp_min{0}, tmp_max{0};
            scalar _min{0}, _max{0};
            for(int i = blockIdx.x * blockDim.x + threadIdx.x;
                i < N;
                i += blockDim.x * gridDim.x)
            {
                _min = min(_min, in[i]);
                _max = max(_max, in[i]);
            }

            tmp_min = _min;
            tmp_max = _max;

            blockReduceMinMax(tmp_min, &_min, &tmp);
            blockReduceMinMax(tmp_max, &tmp, &_max);

            if (threadIdx.x==0)
            {
                out_min[blockIdx.x] = _min;
                out_max[blockIdx.x] = _max;
            }
        }

        std::pair<scalar, scalar> minmax_component(const vectorfield & vf)
        {
            int N = 3*vf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static scalarfield out_min(blocks, 0);
            Vectormath::fill(out_min, 0);
            static scalarfield out_max(blocks, 0);
            Vectormath::fill(out_max, 0);
            static scalarfield temp(1, 0);
            Vectormath::fill(temp, 0);

            cu_MinMax<<<blocks, threads>>>(&vf[0][0], out_min.data(), out_max.data(), N);
            cu_MinMax<<<1, 1024>>>(out_min.data(), out_min.data(), temp.data(), blocks);
            cu_MinMax<<<1, 1024>>>(out_max.data(), temp.data(), out_max.data(), blocks);
            CU_CHECK_AND_SYNC();

            return std::pair<scalar, scalar>{out_min[0], out_max[0]};
        }


        /////////////////////////////////////////////////////////////////

        scalar angle(const Vector3 & v1, const Vector3 & v2)
        {
            scalar cosa = v1.dot(v2);
            return std::acos(cosa);
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


        std::array<scalar, 3> Magnetization(const vectorfield & vf)
        {
            auto M = mean(vf);
            return std::array<scalar,3>{M[0], M[1], M[2]};
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

            const auto & pos = geometry.positions;
            scalar charge = 0;

            // Compute Delaunay for unitcell + basis with neighbouring lattice sites in directions a, b, and a+b
            std::vector<Data::vector2_t> basis_cell_points(geometry.n_cell_atoms + 3);
            for(int i = 0; i < geometry.n_cell_atoms; i++)
            {
                for(int j=0; j<2; j++)
                {
                    basis_cell_points[i].x = double(geometry.cell_atoms[i][j] * geometry.bravais_vectors[j][0]);
                    basis_cell_points[i].y = double(geometry.cell_atoms[i][j] * geometry.bravais_vectors[j][1]);
                }
            }

            Vector3 basis_offset = geometry.cell_atoms[0][0] * geometry.bravais_vectors[0] + geometry.cell_atoms[0][1] * geometry.bravais_vectors[1];

            // a+b
            basis_cell_points[geometry.n_cell_atoms].x   = double((geometry.bravais_vectors[0] + geometry.bravais_vectors[1] + basis_offset)[0]);
            basis_cell_points[geometry.n_cell_atoms].y   = double((geometry.bravais_vectors[0] + geometry.bravais_vectors[1] + basis_offset)[1]);
            // b
            basis_cell_points[geometry.n_cell_atoms+1].x = double((geometry.bravais_vectors[1] + basis_offset)[0]);
            basis_cell_points[geometry.n_cell_atoms+1].y = double((geometry.bravais_vectors[1] + basis_offset)[1]);
            // a
            basis_cell_points[geometry.n_cell_atoms+2].x = double((geometry.bravais_vectors[0] + basis_offset)[0]);
            basis_cell_points[geometry.n_cell_atoms+2].y = double((geometry.bravais_vectors[0] + basis_offset)[1]);

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
        __global__ void cu_transform(const Vector3 * spins, const Vector3 * force, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            Vector3 e1, a2, A;
            scalar detAi;
            if(idx < N)
            {
                e1 = spins[idx];
                A = 0.5 * force[idx];

                // 1/determinant(A)
                detAi = 1.0 / (1 + pow(A.norm(), 2.0));

                // calculate equation without the predictor?
                a2 = e1 - e1.cross(A);

                out[idx][0] = (a2[0] * (A[0] * A[0] + 1   ) + a2[1] * (A[0] * A[1] - A[2]) + a2[2] * (A[0] * A[2] + A[1])) * detAi;
                out[idx][1] = (a2[0] * (A[1] * A[0] + A[2]) + a2[1] * (A[1] * A[1] + 1   ) + a2[2] * (A[1] * A[2] - A[0])) * detAi;
                out[idx][2] = (a2[0] * (A[2] * A[0] - A[1]) + a2[1] * (A[2] * A[1] + A[0]) + a2[2] * (A[2] * A[2] + 1   )) * detAi;
            }
        }
        void transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
        {
            int n = spins.size();
            cu_transform<<<(n+1023)/1024, 1024>>>(spins.data(), force.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        void get_random_vector(std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec)
        {
            for (int dim = 0; dim < 3; ++dim)
            {
                vec[dim] = distribution(prng);
            }
        }

        __global__ void cu_get_random_vectorfield(Vector3 * xi, size_t N)
        {
            unsigned long long subsequence = 0;
            unsigned long long offset= 0;

            curandState_t state;
            for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
                idx < N;
                idx +=  blockDim.x * gridDim.x)
            {
                curand_init(idx,subsequence,offset,&state);
                for (int dim=0;dim<3; ++dim)
                {
                    xi[idx][dim] = llroundf(curand_uniform(&state))*2-1;
                }
            }
        }
        void get_random_vectorfield(std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, vectorfield & xi)
        {
            int n = xi.size();
            cu_get_random_vectorfield<<<(n+1023)/1024, 1024>>>(xi.data(), n);
            CU_CHECK_AND_SYNC();
        }

        void get_random_vector_unitsphere(std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec)
        {
            scalar v_z = distribution(prng);
            scalar phi = distribution(prng);

            scalar r_xy = std::sqrt(1 - v_z*v_z);

            vec[0] = r_xy * std::cos(2*Pi*phi);
            vec[1] = r_xy * std::sin(2*Pi*phi);
            vec[2] = v_z;
        }
        // __global__ void cu_get_random_vectorfield_unitsphere(Vector3 * xi, size_t N)
        // {
        //     unsigned long long subsequence = 0;
        //     unsigned long long offset= 0;

        //     curandState_t state;
        //     for(int idx = blockIdx.x * blockDim.x + threadIdx.x;
        //         idx < N;
        //         idx +=  blockDim.x * gridDim.x)
        //     {
        //         curand_init(idx,subsequence,offset,&state);

        //         scalar v_z = llroundf(curand_uniform(&state))*2-1;
        //         scalar phi = llroundf(curand_uniform(&state))*2-1;

        // 	    scalar r_xy = std::sqrt(1 - v_z*v_z);

        //         xi[idx][0] = r_xy * std::cos(2*Pi*phi);
        //         xi[idx][1] = r_xy * std::sin(2*Pi*phi);
        //         xi[idx][2] = v_z;
        //     }
        // }
        // void get_random_vectorfield_unitsphere(std::mt19937 & prng, vectorfield & xi)
        // {
        //     int n = xi.size();
        //     cu_get_random_vectorfield<<<(n+1023)/1024, 1024>>>(xi.data(), n);
        //     CU_CHECK_AND_SYNC();
        // }
        // The above CUDA implementation does not work correctly.
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
            // Starting value
            fill(distribution, gradient_start);

            // Basic linear gradient distribution
            add_c_dot(gradient_inclination, gradient_direction, geometry.positions, distribution);

            // Get the minimum (i.e. starting point) of the distribution
            scalar bmin = geometry.bounds_min.dot(gradient_direction);
            scalar bmax = geometry.bounds_max.dot(gradient_direction);
            scalar dist_min = std::min(bmin, bmax);
            // Set the starting point
            add(distribution, -dist_min);

            // Cut off negative values
            set_range(distribution, range_min, range_max);
        }


        void directional_gradient(const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions, const Vector3 & direction, vectorfield & gradient)
        {
            // std::cout << "start gradient" << std::endl;
            vectorfield translations = { { 0,0,0 }, { 0,0,0 }, { 0,0,0 } };
            auto& n_cells = geometry.n_cells;

            Vector3 a = geometry.bravais_vectors[0]; // translation vectors of the system
            Vector3 b = geometry.bravais_vectors[1];
            Vector3 c = geometry.bravais_vectors[2];

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

                gradient[ispin].setZero();

                std::vector<Vector3> euclidean { {1,0,0}, {0,1,0}, {0,0,1} };
                std::vector<Vector3> contrib = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
                Vector3 proj = {0, 0, 0};
                Vector3 projection_inv = {0, 0, 0};

                // TODO: both loops together.

                // Loop over neighbours of this vector to calculate contributions of finite differences to current direction
                for(unsigned int j = 0; j < neigh.size(); ++j)
                {
                    if ( boundary_conditions_fulfilled(geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations) )
                    {
                        // Index of neighbour
                        int ineigh = idx_from_translations(n_cells, geometry.n_cell_atoms, translations_i, neigh[j].translations);
                        if (ineigh >= 0)
                        {
                            auto d = geometry.positions[ineigh] - geometry.positions[ispin];
                            for (int dim=0; dim<3; ++dim)
                            {
                                proj[dim] += std::abs(euclidean[dim].dot(d.normalized()));
                            }
                        }
                    }
                }
                for (int dim=0; dim<3; ++dim)
                {
                    if (std::abs(proj[dim]) > 1e-10)
                        projection_inv[dim] = 1.0/proj[dim];
                }
                // Loop over neighbours of this vector to calculate finite differences
                for(unsigned int j = 0; j < neigh.size(); ++j)
                {
                    if ( boundary_conditions_fulfilled(geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations) )
                    {
                        // Index of neighbour
                        int ineigh = idx_from_translations(n_cells, geometry.n_cell_atoms, translations_i, neigh[j].translations);
                        if (ineigh >= 0)
                        {
                            auto d = geometry.positions[ineigh] - geometry.positions[ispin];
                            for (int dim=0; dim<3; ++dim)
                            {
                                contrib[dim] += euclidean[dim].dot(d) / d.dot(d) * ( vf[ineigh] - vf[ispin] );
                            }
                        }
                    }
                }

                for (int dim=0; dim<3; ++dim)
                {
                    gradient[ispin] += direction[dim]*projection_inv[dim] * contrib[dim];
                }
            }
        }



        /////////////////////////////////////////////////////////////////

        vectorfield change_dimensions(vectorfield & sf, int n_cell_atoms, intfield n_cells,
            intfield dimensions_new, std::array<int,3> shift)
        {
            int N_old = n_cell_atoms*n_cells[0]*n_cells[1]*dimensions_new[2];
            int N_new = n_cell_atoms*dimensions_new[0]*dimensions_new[1]*dimensions_new[2];
            vectorfield newfield(N_new);

            for (int i=0; i<dimensions_new[0]; ++i)
            {
                for (int j=0; j<dimensions_new[1]; ++j)
                {
                    for (int k=0; k<dimensions_new[2]; ++k)
                    {
                        for (int iatom=0; iatom<n_cell_atoms; ++iatom)
                        {
                            int idx_old = iatom + idx_from_translations(n_cells, n_cell_atoms, {i,j,k});

                            int idx_new = iatom + idx_from_translations(dimensions_new, n_cell_atoms, {i,j,k}, shift.data());

                            if ( (i>=n_cells[0]) || (j>=n_cells[1]) || (k>=n_cells[2]))
                                newfield[idx_new] = {0,0,1};
                            else
                                newfield[idx_new] = sf[idx_old];
                        }
                    }
                }
            }
            return newfield;
        }

        /////////////////////////////////////////////////////////////////


        __global__ void cu_fill(scalar *sf, scalar s, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                sf[idx] = s;
            }
        }
        void fill(scalarfield & sf, scalar s)
        {
            int n = sf.size();
            cu_fill<<<(n+1023)/1024, 1024>>>(sf.data(), s, n);
            CU_CHECK_AND_SYNC();
        }
        __global__ void cu_fill_mask(scalar *sf, scalar s, const int * mask, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                sf[idx] = mask[idx]*s;
            }
        }
        void fill(scalarfield & sf, scalar s, const intfield & mask)
        {
            int n = sf.size();
            cu_fill_mask<<<(n+1023)/1024, 1024>>>(sf.data(), s, mask.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_scale(scalar *sf, scalar s, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                sf[idx] *= s;
            }
        }
        void scale(scalarfield & sf, scalar s)
        {
            int n = sf.size();
            cu_scale<<<(n+1023)/1024, 1024>>>(sf.data(), s, n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_add(scalar *sf, scalar s, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                sf[idx] += s;
            }
        }
        void add(scalarfield & sf, scalar s)
        {
            int n = sf.size();
            cu_add<<<(n+1023)/1024, 1024>>>(sf.data(), s, n);
            cudaDeviceSynchronize();
        }

        scalar sum(const scalarfield & sf)
        {
            int N = sf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static scalarfield ret(1, 0);
            Vectormath::fill(ret, 0);
            cu_sum<<<blocks, threads>>>(sf.data(), ret.data(), N);
            CU_CHECK_AND_SYNC();
            return ret[0];
        }

        scalar mean(const scalarfield & sf)
        {
            int N = sf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static scalarfield ret(1, 0);
            Vectormath::fill(ret, 0);

            cu_sum<<<blocks, threads>>>(sf.data(), ret.data(), N);
            CU_CHECK_AND_SYNC();

            ret[0] = ret[0]/N;
            return ret[0];
        }

        __global__ void cu_divide(const scalar * numerator, const scalar * denominator, scalar * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += numerator[idx] / denominator[idx];
            }
        }
        void divide( const scalarfield & numerator, const scalarfield & denominator, scalarfield & out )
        {
            int n = numerator.size();
            cu_divide<<<(n+1023)/1024, 1024>>>(numerator.data(), denominator.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        void set_range(scalarfield & sf, scalar sf_min, scalar sf_max)
        {
            #pragma omp parallel for
            for (unsigned int i = 0; i<sf.size(); ++i)
                sf[i] = std::min( std::max( sf_min, sf[i] ), sf_max );
        }

        __global__ void cu_fill(Vector3 *vf1, Vector3 v2, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                vf1[idx] = v2;
            }
        }
        void fill(vectorfield & vf, const Vector3 & v)
        {
            int n = vf.size();
            cu_fill<<<(n+1023)/1024, 1024>>>(vf.data(), v, n);
            CU_CHECK_AND_SYNC();
        }
        __global__ void cu_fill_mask(Vector3 *vf1, Vector3 v2, const int * mask, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                vf1[idx] = v2;
            }
        }
        void fill(vectorfield & vf, const Vector3 & v, const intfield & mask)
        {
            int n = vf.size();
            cu_fill_mask<<<(n+1023)/1024, 1024>>>(vf.data(), v, mask.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_normalize_vectors(Vector3 *vf, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                vf[idx].normalize();
            }
        }
        void normalize_vectors(vectorfield & vf)
        {
            int n = vf.size();
            cu_normalize_vectors<<<(n+1023)/1024, 1024>>>(vf.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_norm(const Vector3 * vf, scalar * norm, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                norm[idx] = vf[idx].norm();
            }
        }
        void norm( const vectorfield & vf, scalarfield & norm )
        {
            int n = vf.size();
            cu_norm<<<(n+1023)/1024, 1024>>>(vf.data(), norm.data(), n);
            CU_CHECK_AND_SYNC();
        }

        scalar max_abs_component(const vectorfield & vf)
        {
            // We want the Maximum of Absolute Values of all force components on all images
            // Find minimum and maximum values
            std::pair<scalar,scalar> minmax = minmax_component(vf);
            scalar absmin = std::abs(minmax.first);
            scalar absmax = std::abs(minmax.second);
            // Maximum of absolute values
            return std::max(absmin, absmax);
        }

        __global__ void cu_scale(Vector3 *vf1, scalar sc, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                vf1[idx] *= sc;
            }
        }
        void scale(vectorfield & vf, const scalar & sc)
        {
            int n = vf.size();
            cu_scale<<<(n+1023)/1024, 1024>>>(vf.data(), sc, n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_scale(Vector3 *vf1, const scalar * sf, bool inverse, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                if( inverse )
                    vf1[idx] /= sf[idx];
                else
                    vf1[idx] *= sf[idx];
            }
        }
        void scale(vectorfield & vf, const scalarfield & sf, bool inverse)
        {
            int n = vf.size();
            cu_scale<<<(n+1023)/1024, 1024>>>(vf.data(), sf.data(), inverse, n);
            CU_CHECK_AND_SYNC();
        }

        Vector3 sum(const vectorfield & vf)
        {
            int N = vf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static vectorfield ret(1, {0,0,0});
            Vectormath::fill(ret, {0,0,0});
            cu_sum<<<blocks, threads>>>(vf.data(), ret.data(), N);
            CU_CHECK_AND_SYNC();
            return ret[0];
        }

        Vector3 mean(const vectorfield & vf)
        {
            int N = vf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static vectorfield ret(1, {0,0,0});
            Vectormath::fill(ret, {0,0,0});

            cu_sum<<<blocks, threads>>>(vf.data(), ret.data(), N);
            CU_CHECK_AND_SYNC();

            ret[0] = ret[0]/N;
            return ret[0];
        }




        __global__ void cu_dot(const Vector3 *vf1, const Vector3 *vf2, scalar *out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = vf1[idx].dot(vf2[idx]);
            }
        }

        scalar dot(const vectorfield & vf1, const vectorfield & vf2)
        {
            int n = vf1.size();
            static scalarfield sf(n, 0);
            Vectormath::fill(sf, 0);
            scalar ret;

            // Dot product
            cu_dot<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), sf.data(), n);
            CU_CHECK_AND_SYNC();

            // reduction
            ret = sum(sf);
            return ret;
        }

        void dot(const vectorfield & vf1, const vectorfield & vf2, scalarfield & s)
        {
            int n = vf1.size();

            // Dot product
            cu_dot<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), s.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_scalardot(const scalar * s1, const scalar * s2, scalar * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = s1[idx] * s2[idx];
            }
        }
        // computes the product of scalars in s1 and s2
        // s1 and s2 are scalarfields
        void dot( const scalarfield & s1, const scalarfield & s2, scalarfield & out )
        {
            int n = s1.size();

            // Dot product
            cu_scalardot<<<(n+1023)/1024, 1024>>>(s1.data(), s2.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_cross(const Vector3 *vf1, const Vector3 *vf2, Vector3 *out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = vf1[idx].cross(vf2[idx]);
            }
        }
        // The wrapper for the calling of the actual kernel
        void cross(const vectorfield & vf1, const vectorfield & vf2, vectorfield & s)
        {
            int n = vf1.size();

            // Dot product
            cu_cross<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), s.data(), n);
            CU_CHECK_AND_SYNC();
        }




        __global__ void cu_add_c_a(scalar c, Vector3 a, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*a;
            }
        }
        // out[i] += c*a
        void add_c_a(const scalar & c, const Vector3 & a, vectorfield & out)
        {
            int n = out.size();
            cu_add_c_a<<<(n+1023)/1024, 1024>>>(c, a, out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_add_c_a2(scalar c, const Vector3 * a, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*a[idx];
            }
        }
        // out[i] += c*a[i]
        void add_c_a(const scalar & c, const vectorfield & a, vectorfield & out)
        {
            int n = out.size();
            cu_add_c_a2<<<(n+1023)/1024, 1024>>>(c, a.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_add_c_a2_mask(scalar c, const Vector3 * a, Vector3 * out, const int * mask, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*mask[idx]*a[idx];
            }
        }
        // out[i] += c*a[i]
        void add_c_a(const scalar & c, const vectorfield & a, vectorfield & out, const intfield & mask)
        {
            int n = out.size();
            cu_add_c_a2_mask<<<(n+1023)/1024, 1024>>>(c, a.data(), out.data(), mask.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_add_c_a3(const scalar * c, const Vector3 * a, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c[idx]*a[idx];
            }
        }
        // out[i] += c[i]*a[i]
        void add_c_a( const scalarfield & c, const vectorfield & a, vectorfield & out )
        {
            int n = out.size();
            cu_add_c_a3<<<(n+1023)/1024, 1024>>>(c.data(), a.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }


        __global__ void cu_set_c_a(scalar c, Vector3 a, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c*a;
            }
        }
        // out[i] = c*a
        void set_c_a(const scalar & c, const Vector3 & a, vectorfield & out)
        {
            int n = out.size();
            cu_set_c_a<<<(n+1023)/1024, 1024>>>(c, a, out.data(), n);
            CU_CHECK_AND_SYNC();
        }
        __global__ void cu_set_c_a_mask(scalar c, Vector3 a, Vector3 * out, const int * mask, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = mask[idx]*c*a;
            }
        }
        // out[i] = c*a
        void set_c_a(const scalar & c, const Vector3 & a, vectorfield & out, const intfield & mask)
        {
            int n = out.size();
            cu_set_c_a_mask<<<(n+1023)/1024, 1024>>>(c, a, out.data(), mask.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_set_c_a2(scalar c, const Vector3 * a, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c*a[idx];
            }
        }
        // out[i] = c*a[i]
        void set_c_a(const scalar & c, const vectorfield & a, vectorfield & out)
        {
            int n = out.size();
            cu_set_c_a2<<<(n+1023)/1024, 1024>>>(c, a.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }
        __global__ void cu_set_c_a2_mask(scalar c, const Vector3 * a, Vector3 * out, const int * mask, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = mask[idx]*c*a[idx];
            }
        }
        // out[i] = c*a[i]
        void set_c_a(const scalar & c, const vectorfield & a, vectorfield & out, const intfield & mask)
        {
            int n = out.size();
            cu_set_c_a2_mask<<<(n+1023)/1024, 1024>>>(c, a.data(), out.data(), mask.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_set_c_a3(const scalar * c, const Vector3 * a, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c[idx]*a[idx];
            }
        }
        // out[i] = c[i]*a[i]
        void set_c_a( const scalarfield & c, const vectorfield & a, vectorfield & out )
        {
            int n = out.size();
            cu_set_c_a3<<<(n+1023)/1024, 1024>>>(c.data(), a.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }



        __global__ void cu_add_c_dot(scalar c, Vector3 a, const Vector3 * b, scalar * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*a.dot(b[idx]);
            }
        }
        // out[i] += c * a*b[i]
        void add_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out)
        {
            int n = out.size();
            cu_add_c_dot<<<(n+1023)/1024, 1024>>>(c, a, b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_add_c_dot(scalar c, const Vector3 * a, const Vector3 * b, scalar * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*a[idx].dot(b[idx]);
            }
        }
        // out[i] += c * a[i]*b[i]
        void add_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out)
        {
            int n = out.size();
            cu_add_c_dot<<<(n+1023)/1024, 1024>>>(c, a.data(), b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }


        __global__ void cu_set_c_dot(scalar c, Vector3 a, const Vector3 * b, scalar * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c*a.dot(b[idx]);
            }
        }
        // out[i] = c * a*b[i]
        void set_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out)
        {
            int n = out.size();
            cu_set_c_dot<<<(n+1023)/1024, 1024>>>(c, a, b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        __global__ void cu_set_c_dot(scalar c, const Vector3 * a, const Vector3 * b, scalar * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c*a[idx].dot(b[idx]);
            }
        }
        // out[i] = c * a[i]*b[i]
        void set_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out)
        {
            int n = out.size();
            cu_set_c_dot<<<(n+1023)/1024, 1024>>>(c, a.data(), b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }



        // out[i] += c * a x b[i]
        __global__ void cu_add_c_cross(scalar c, const Vector3 a, const Vector3 * b, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*a.cross(b[idx]);
            }
        }
        void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
        {
            int n = out.size();
            cu_add_c_cross<<<(n+1023)/1024, 1024>>>(c, a, b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        // out[i] += c * a[i] x b[i]
        __global__ void cu_add_c_cross(scalar c, const Vector3 * a, const Vector3 * b, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c*a[idx].cross(b[idx]);
            }
        }
        void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
            int n = out.size();
            cu_add_c_cross<<<(n+1023)/1024, 1024>>>(c, a.data(), b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        // out[i] += c * a[i] x b[i]
        __global__ void cu_add_c_cross(const scalar * c, const Vector3 * a, const Vector3 * b, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] += c[idx]*a[idx].cross(b[idx]);
            }
        }
        void add_c_cross(const scalarfield & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
            int n = out.size();
            cu_add_c_cross<<<(n+1023)/1024, 1024>>>(c.data(), a.data(), b.data(), out.data(), n);
            cudaDeviceSynchronize();
        }


        // out[i] = c * a x b[i]
        __global__ void cu_set_c_cross(scalar c, const Vector3 a, const Vector3 * b, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c*a.cross(b[idx]);
            }
        }
        void set_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
        {
            int n = out.size();
            cu_set_c_cross<<<(n+1023)/1024, 1024>>>(c, a, b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }

        // out[i] = c * a[i] x b[i]
        __global__ void cu_set_c_cross(scalar c, const Vector3 * a, const Vector3 * b, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                out[idx] = c*a[idx].cross(b[idx]);
            }
        }
        void set_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
            int n = out.size();
            cu_set_c_cross<<<(n+1023)/1024, 1024>>>(c, a.data(), b.data(), out.data(), n);
            CU_CHECK_AND_SYNC();
        }
    }
}

#endif