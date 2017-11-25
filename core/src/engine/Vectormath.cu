#ifdef USE_CUDA

#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>
#include <algorithm>

#include <curand.h>
#include <curand_kernel.h>

using namespace Utility;

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
            cudaDeviceSynchronize();

            return std::pair<scalar, scalar>{out_min[0], out_max[0]};
		}


		/////////////////////////////////////////////////////////////////


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


		void Build_Spins(vectorfield & spin_pos, const std::vector<Vector3> & basis_atoms, 
            const std::vector<Vector3> & translation_vectors, const intfield & n_cells)
        {
            // Check for erronous input placing two spins on the same location
            int max_a = std::min(10, n_cells[0]);
            int max_b = std::min(10, n_cells[1]);
            int max_c = std::min(10, n_cells[2]);
            Vector3 sp;
            for (unsigned int i = 0; i < basis_atoms.size(); ++i)
            {
                for (unsigned int j = 0; j < basis_atoms.size(); ++j)
                {
                    for (int ka = -max_a; ka <= max_a; ++ka)
                    {
                        for (int k2 = -max_b; k2 <= max_b; ++k2)
                        {
                            for (int k3 = -max_c; k3 <= max_c; ++k3)
                            {
                                // Norm is zero if translated basis atom is at position of another basis atom
                                sp = basis_atoms[i] - (basis_atoms[j]
                                    + ka * translation_vectors[0] + k2 * translation_vectors[1] + 
                                    k3 * translation_vectors[2]);
                                if ( (i != j || ka != 0 || k2 != 0 || k3 != 0) && 
                                    std::abs(sp[0]) < 1e-9 && std::abs(sp[1]) < 1e-9 &&
                                    std::abs(sp[2]) < 1e-9 )
                                {
                                    spirit_throw(Exception_Classifier::System_not_Initialized, Log_Level::Severe,
                                        "Unable to initialize Spin-System, since 2 spins occupy the same space.\nPlease check the config file!");
                                }
                            }
                        }
                    }
                }
            }

            // Build up the spins array
            int i, j, k, s, ispin;
            int nos_basic = basis_atoms.size();
            //int nos = nos_basic * n_cells[0] * n_cells[1] * n_cells[2];
            Vector3 build_array;
            for (k = 0; k < n_cells[2]; ++k) {
                for (j = 0; j < n_cells[1]; ++j) {
                    for (i = 0; i < n_cells[0]; ++i) {
                        for (s = 0; s < nos_basic; ++s) {
                            ispin = k * n_cells[1] * n_cells[0] * nos_basic + 
                                    j * n_cells[0] * nos_basic + i * nos_basic + s;
                            build_array = i * translation_vectors[0] + j * translation_vectors[1] + 
                                            k * translation_vectors[2];
                            // paste initial spin orientations across the lattice translations
                            //spins[dim*nos + ispin] = spins[dim*nos + s];
                            // calculate the spin positions
                            spin_pos[ispin] = basis_atoms[s] + build_array;
                        }// endfor s
                    }// endfor k
                }// endfor j
            }// endfor dim

        }// end Build_Spins


		std::array<scalar, 3> Magnetization(const vectorfield & vf)
		{
            
            auto M = mean(vf);
            cudaDeviceSynchronize();
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

        scalar TopologicalCharge(const vectorfield & vf, const vectorfield & vf_pos, const std::vector<std::array<int, 3>> & triangulation)
        {
            // TODO: this still ignores periodical boundaries, as they are not part of the delaunay triangulation!

            scalar charge = 0, sign;
            Vector3 triangle_normal;
            for (int i = 0; i < triangulation.size(); ++i)
            {
                int i1 = triangulation[i][0];
                int i2 = triangulation[i][1];
                int i3 = triangulation[i][2];

                auto& vp1 = vf_pos[i1];
                auto& vp2 = vf_pos[i2];
                auto& vp3 = vf_pos[i3];

                // TODO: this will only work if the vf_pos are in the xy-plane!
                triangle_normal = (vp1-vp2).cross(vp1-vp3);
                triangle_normal.normalize();
                sign = triangle_normal[2]/std::abs(triangle_normal[2]);

                auto& v1 = vf[i1];
                auto& v2 = vf[i2];
                auto& v3 = vf[i3];

                // charge += sign * solid_angle_1(v1, v2, v3);
                charge += sign * solid_angle_2(v1, v2, v3);
            }
            return charge / (4*M_PI);
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
		}

        void get_random_vector_unitsphere(std::uniform_real_distribution<scalar> & distribution, std::mt19937 & prng, Vector3 & vec)
        {
			scalar v_z = distribution(prng);
			scalar phi = distribution(prng);

			scalar r_xy = std::sqrt(1 - v_z*v_z);

			vec[0] = r_xy * std::cos(2*M_PI*phi);
			vec[1] = r_xy * std::sin(2 * M_PI*phi);
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

        //         xi[idx][0] = r_xy * std::cos(2*M_PI*phi);
        //         xi[idx][1] = r_xy * std::sin(2 * M_PI*phi);
        //         xi[idx][2] = v_z;
        //     }
        // }
        // void get_random_vectorfield_unitsphere(std::mt19937 & prng, vectorfield & xi)
        // {
        //     int n = xi.size();
        //     cu_get_random_vectorfield<<<(n+1023)/1024, 1024>>>(xi.data(), n);
        //     cudaDeviceSynchronize();
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


        void directional_gradient(const vectorfield & vf, const Data::Geometry & geometry, const intfield & boundary_conditions, const Vector3 & direction, vectorfield & gradient)
        {
            // std::cout << "start gradient" << std::endl;
            vectorfield translations = { { 0,0,0 }, { 0,0,0 }, { 0,0,0 } };
            auto& n_cells = geometry.n_cells;

            Vector3 a = geometry.translation_vectors[0]; // translation vectors of the system
            Vector3 b = geometry.translation_vectors[1];
            Vector3 c = geometry.translation_vectors[2];

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

            // difference quotients in different directions
            Vector3 diffq, diffqx, diffqy, diffqz;

            // Loop over vectorfield
            int nos = vf.size();
            for(unsigned int ispin = 0; ispin < nos; ++ispin)
            {
                // auto translations_i = translations_from_idx(n_cells, geometry.n_spins_basic_domain, ispin); // transVec of spin i
                // int k = i%geometry.n_spins_basic_domain; // index within unit cell - k=0 for all cases used in the thesis
                scalar n = 0;

                diffqx = { 0,0,0 }; diffqy = { 0,0,0 }; diffqz = { 0,0,0 };
                
                for(unsigned int j = 0; j < neigh.size(); ++j)
                {
                    int jspin = idx_from_pair(ispin, boundary_conditions, n_cells, geometry.n_spins_basic_domain, geometry.atom_types, neigh[j]);
                    if (jspin >= 0)
                    // if ( boundary_conditions_fulfilled(geometry.n_cells, boundary_conditions, translations_i, neigh[j].translations) )
                    {
                        // index of neighbour
                        // int ineigh = idx_from_translations(n_cells, geometry.n_spins_basic_domain, translations_i, neigh[j].translations);
                        
                        Vector3 translationVec3 = neigh[j].translations[0]*a + neigh[j].translations[1]*b + neigh[j].translations[2]*c;
                        // add "+ geometry.basis_atoms[neigh[k][j].jatom] - geometry.basis_atoms[k]" for unit cells with >1atom ?

                        // difference quotient in direction of the neighbour
                        diffq = ( vf[jspin] - vf[ispin] ) / translationVec3.norm();

                        // projection of difference quotient in euclidian space
                        diffqx += translationVec3[0]*diffq;
                        diffqy += translationVec3[1]*diffq;
                        diffqz += translationVec3[2]*diffq;
                        
                        // boundary conditions considered
                        n += 1;
                    }
                }

                diffqx = diffqx/n; diffqy = diffqy/n; diffqz = diffqz/n;


                gradient.push_back(direction[0]*diffqx + direction[1]*diffqy + direction[2]*diffqz); // dot(direction, diffqxyz, scalarfield & out)
            }
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
        }

        scalar  max_abs_component(const vectorfield & vf)
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
            cudaDeviceSynchronize();
        }

        Vector3 sum(const vectorfield & vf)
		{
            int N = vf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static vectorfield ret(1, {0,0,0});
            Vectormath::fill(ret, {0,0,0});
            cu_sum<<<blocks, threads>>>(vf.data(), ret.data(), N);
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();

            // reduction
            ret = sum(sf);
            return ret;
        }

        void dot(const vectorfield & vf1, const vectorfield & vf2, scalarfield & s)
        {
            int n = vf1.size();

            // Dot product
            cu_dot<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), s.data(), n);
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
        }
    }
}

#endif