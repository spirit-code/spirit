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
                val  = min(val,  __shfl_down(val, offset));
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
            scalar min, max;
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                scalar val = in[idx];
                blockReduceMinMax(val, &min, &max);

                if (threadIdx.x==0)
                {
                    out_min[blockIdx.x] = min;
                    out_max[blockIdx.x] = max;
                }
            }
        }

        std::pair<scalar, scalar> minmax_component(const vectorfield & vf)
		{
            int N = 3*vf.size();
            int threads = 512;
            int blocks = min((N + threads - 1) / threads, 1024);

            static scalarfield out_min(blocks);
            static scalarfield out_max(blocks);
            static scalarfield temp(1);

            cu_MinMax<<<blocks, threads>>>(&vf[0][0], out_min.data(), out_max.data(), N);
            cu_MinMax<<<1, 1024>>>(out_min.data(), out_min.data(), temp.data(), blocks);
            cu_MinMax<<<1, 1024>>>(out_max.data(), temp.data(), out_max.data(), blocks);
            cudaDeviceSynchronize();

            return std::pair<scalar, scalar>{out_min[0], out_max[0]};
		}


		/////////////////////////////////////////////////////////////////


		void rotate(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out)
		{
			v_out = v * std::cos(angle) + axis.cross(v) * std::sin(angle);
		}

		Vector3 decompose(const Vector3 & v, const std::vector<Vector3> & basis)
		{
			Eigen::Ref<const Matrix3> A = Eigen::Map<const Matrix3>(basis[0].data());
			return A.colPivHouseholderQr().solve(v);
		}


		/////////////////////////////////////////////////////////////////


		void Build_Spins(vectorfield & spin_pos, const std::vector<Vector3> & basis_atoms, const std::vector<Vector3> & translation_vectors, const std::vector<int> & n_cells)
		{
			// Check for erronous input placing two spins on the same location
			Vector3 sp;
			for (unsigned int i = 0; i < basis_atoms.size(); ++i)
			{
				for (unsigned int j = 0; j < basis_atoms.size(); ++j)
				{
					for (int k1 = -2; k1 <= 2; ++k1)
					{
						for (int k2 = -2; k2 <= 2; ++k2)
						{
							for (int k3 = -2; k3 <= 2; ++k3)
							{
								// Norm is zero if translated basis atom is at position of another basis atom
								sp = basis_atoms[i] - (basis_atoms[j]
									+ k1*translation_vectors[0] + k2*translation_vectors[1] + k3*translation_vectors[2]);
								if ((i != j || k1 != 0 || k2 != 0 || k3 != 0) && std::abs(sp[0]) < 1e-9 && std::abs(sp[1]) < 1e-9 && std::abs(sp[2]) < 1e-9)
								{
									Log(Utility::Log_Level::Severe, Utility::Log_Sender::All, "Unable to initialize Spin-System, since 2 spins occupy the same space.\nPlease check the config file!");
									Log.Append_to_File();
									throw Utility::Exception::System_not_Initialized;
								}
							}
						}
					}
				}
			}

			// Build up the spins array
			int i, j, k, s, pos;
			int nos_basic = basis_atoms.size();
			//int nos = nos_basic * n_cells[0] * n_cells[1] * n_cells[2];
			Vector3 build_array;
			for (k = 0; k < n_cells[2]; ++k) {
				for (j = 0; j < n_cells[1]; ++j) {
					for (i = 0; i < n_cells[0]; ++i) {
						for (s = 0; s < nos_basic; ++s) {
							pos = k*n_cells[1] * n_cells[0] * nos_basic + j*n_cells[0] * nos_basic + i*nos_basic + s;
							build_array = i*translation_vectors[0] + j*translation_vectors[1] + k*translation_vectors[2];
							// paste initial spin orientations across the lattice translations
							//spins[dim*nos + pos] = spins[dim*nos + s];
							// calculate the spin positions
							spin_pos[pos] = basis_atoms[s] + build_array;
						}// endfor s
					}// endfor k
				}// endfor j
			}// endfor dim

		};// end Build_Spins


		std::array<scalar, 3> Magnetization(const vectorfield & vf)
		{
            
            auto M = mean(vf);
            cudaDeviceSynchronize();
            return std::array<scalar,3>{M[0], M[1], M[2]};
		}

		scalar TopologicalCharge(const vectorfield & vf)
		{
        	Log(Utility::Log_Level::Warning, Utility::Log_Sender::All, std::string("Calculating the topological charge is not yet implemented"));
			return 0;
		}

        // Utility function for the SIB Optimizer
        __global__ void cu_transform(const Vector3 * spins, const Vector3 * force, Vector3 * out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
			Vector3 e1, a2, A;
			scalar detAi;
            if(idx < N)
            {
                e1 = spins[idx];
				A = force[idx];

				// 1/determinant(A)
				detAi = 1.0 / (1 + pow(A.norm(), 2.0));

				// calculate equation without the predictor?
				a2 = e1 + e1.cross(A);

				out[idx][0] = (a2[0] * (1 + A[0] * A[0])    + a2[1] * (A[0] * A[1] + A[2]) + a2[2] * (A[0] * A[2] - A[1]))*detAi;
				out[idx][1] = (a2[0] * (A[1] * A[0] - A[2]) + a2[1] * (1 + A[1] * A[1])    + a2[2] * (A[1] * A[2] + A[0]))*detAi;
				out[idx][2] = (a2[0] * (A[2] * A[0] + A[1]) + a2[1] * (A[2] * A[1] - A[0]) + a2[2] * (1 + A[2] * A[2]))*detAi;
            }
        }
		void transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
		{
            int n = spins.size();
            cu_transform<<<(n+1023)/1024, 1024>>>(spins.data(), force.data(), out.data(), n);
            cudaDeviceSynchronize();
		}

        // Utility function for the SIB Optimizer
        __global__ void cu_get_random_vectorfield(scalar epsilon, Vector3 * xi, size_t N)
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
                    xi[idx][dim] = epsilon*(llroundf(curand_uniform(&state))*2-1);
                }
            }
        }
		void get_random_vectorfield(const Data::Spin_System & sys, scalar epsilon, vectorfield & xi)
		{
            int n = xi.size();
            cu_get_random_vectorfield<<<(n+1023)/1024, 1024>>>(epsilon, xi.data(), n);
            cudaDeviceSynchronize();
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
            cu_sum<<<blocks, threads>>>(sf.data(), ret.data(), N);
            cudaDeviceSynchronize();
            ret[0] = ret[0]/N;
            return ret[0];
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
            static scalarfield sf(n);
            scalar ret;

            // Dot product
            cu_dot<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), sf.data(), n);
            cudaDeviceSynchronize();

            // reduction
            ret = sum(sf);
            return ret;
        }


        // The wrapper for the calling of the actual kernel
        void dot(const vectorfield & vf1, const vectorfield & vf2, scalarfield & s)
        {
            int n = vf1.size();

            // Dot product
            cu_dot<<<(n+1023)/1024, 1024>>>(vf1.data(), vf2.data(), s.data(), n);
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
            cu_add_c_cross<<<(n+1023)/1024, 1024>>>(c, a, b.data(), out.data(), n);
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
            cu_add_c_cross<<<(n+1023)/1024, 1024>>>(c, a.data(), b.data(), out.data(), n);
            cudaDeviceSynchronize();
        }
    }
}

#endif