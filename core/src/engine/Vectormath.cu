#ifdef USE_CUDA

#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>

#include <curand.h>
#include <curand_kernel.h>

// CUDA Version
namespace Engine
{
	namespace Vectormath
	{
		// Returns the Bohr Magneton [meV / T]
		scalar MuB()
		{
			return 0.057883817555;
		}

		// Returns the Boltzmann constant [meV / K]
		scalar kB()
		{
			return 0.08617330350;
		}


		/////////////////////////////////////////////////////////////////


		void rotate(const Vector3 & v, const Vector3 & axis, const scalar & angle, Vector3 & v_out)
		{
			v_out = v * std::cos(angle) + axis.cross(v) * std::sin(angle);
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


        __global__ void cu_Magnetization(const Vector3 *vf, scalar * M, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                atomicAdd(&M[0], vf[idx][0]);
                atomicAdd(&M[1], vf[idx][1]);
                atomicAdd(&M[2], vf[idx][2]);
            }
        }
		std::array<scalar, 3> Magnetization(const vectorfield & vf)
		{
            scalarfield M(3, 0);
            int n = vf.size();
            cu_Magnetization<<<(n+1023)/1024, 1024>>>(vf.data(), M.data(), n);
            cudaDeviceSynchronize();
            
			scalar scale = 1/(scalar)n;
            M[0] *= scale; M[1] *= scale; M[2] *= scale;
            return std::array<scalar,3>{M[0], M[1], M[2]};
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
                atomicExch(&sf[idx], s*sf[idx]);
            }
        }
		void scale(scalarfield & sf, scalar s)
		{
            int n = sf.size();
            cu_scale<<<(n+1023)/1024, 1024>>>(sf.data(), s, n);
            cudaDeviceSynchronize();
		}

        __global__ void cu_sum(const scalar *sf, scalar *out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                atomicAdd(out, sf[idx]);
            }
        }
		scalar sum(const scalarfield & sf)
		{
            scalarfield ret(1, 0);
            int n = sf.size();
            cu_sum<<<(n+1023)/1024, 1024>>>(sf.data(), ret.data(), n);
            cudaDeviceSynchronize();
            return ret[0];
		}

		scalar mean(const scalarfield & sf)
		{
            scalarfield ret(1, 0);
            int n = sf.size();
            cu_sum<<<(n+1023)/1024, 1024>>>(sf.data(), ret.data(), n);
            cudaDeviceSynchronize();
            ret[0] = ret[0]/n;
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

        __global__ void cu_sum(const Vector3 *sf, Vector3 *out, size_t N)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < N)
            {
                atomicAdd(&out[0][0], sf[idx][0]);
                atomicAdd(&out[0][1], sf[idx][1]);
                atomicAdd(&out[0][2], sf[idx][2]);
            }
        }
        Vector3 sum(const vectorfield & vf)
		{
			Vector3 ret = { 0,0,0 };
            int n = vf.size();
            cu_sum<<<(n+1023)/1024, 1024>>>(vf.data(), &ret, n);
            cudaDeviceSynchronize();
			return ret;
		}

		Vector3 mean(const vectorfield & vf)
		{
			Vector3 ret = { 0,0,0 };
            int n = vf.size();
            cu_sum<<<(n+1023)/1024, 1024>>>(vf.data(), &ret, n);
            cudaDeviceSynchronize();
            ret /= (scalar)n;
			return ret;
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
            scalarfield sf(n);
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