#ifdef USE_CUDA

#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>

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


		void Build_Spins(vectorfield & spin_pos, std::vector<Vector3> & basis_atoms, std::vector<Vector3> & translation_vectors, std::vector<int> & n_cells, const int nos_basic)
		{
			Vector3 a = translation_vectors[0];
			Vector3 b = translation_vectors[1];
			Vector3 c = translation_vectors[2];

			int i, j, k, s, pos, nos = nos_basic * n_cells[0] * n_cells[1] * n_cells[2];
			Vector3 build_array;
			for (k = 0; k < n_cells[2]; ++k) {
				for (j = 0; j < n_cells[1]; ++j) {
					for (i = 0; i < n_cells[0]; ++i) {
						for (s = 0; s < nos_basic; ++s) {
							pos = k*n_cells[1] * n_cells[0] * nos_basic + j*n_cells[0] * nos_basic + i*nos_basic + s;
							build_array = i*a + j*b + k*c;
							// paste initial spin orientations across the lattice translations
							//spins[dim*nos + pos] = spins[dim*nos + s];
							// calculate the spin positions
							spin_pos[pos] = basis_atoms[s] + build_array;
						}// endfor s
					}// endfor k
				}// endfor j
			}// endfor dim

			 // Check for erronous input placing two spins on the same location
			Vector3 sp;
			for (unsigned int i = 0; i < basis_atoms.size(); ++i)
			{
				for (unsigned int j = 0; j < basis_atoms.size(); ++j)
				{
					for (int k1 = -2; k1 < 3; ++k1)
					{
						for (int k2 = -2; k2 < 3; ++k2)
						{
							for (int k3 = -2; k3 < 3; ++k3)
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
		};// end Build_Spins


		std::array<scalar, 3> Magnetization(const vectorfield & vf)
		{
			std::array<scalar, 3> M{ 0, 0, 0 };
			int nos = vf.size();
			scalar scale = 1 / (scalar)nos;
			for (int i = 0; i<nos; ++i)
			{
				M[0] += vf[i][0] * scale;
				M[1] += vf[i][1] * scale;
				M[2] += vf[i][2] * scale;
			}
			return M;
		}



		/////////////////////////////////////////////////////////////////

        void fill(scalarfield & sf, scalar s)
		{
			for (unsigned int i = 0; i<sf.size(); ++i)
			{
				sf[i] = s;
			}
		}

		void scale(scalarfield & sf, scalar s)
		{
			for (unsigned int i = 0; i<sf.size(); ++i)
			{
				sf[i] *= s;
			}
		}

		scalar sum(const scalarfield & sf)
		{
			scalar ret = 0;
			for (unsigned int i = 0; i<sf.size(); ++i)
			{
				ret += sf[i];
			}
			return ret;
		}

		scalar mean(const scalarfield & sf)
		{
			scalar ret = 0;
			for (unsigned int i = 0; i<sf.size(); ++i)
			{
				ret = (i - 1) / i * ret + sf[i] / i;
			}
			return ret;
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
			Vector3 ret = { 0,0,0 };
			for (unsigned int i = 0; i<vf.size(); ++i)
			{
				ret += vf[i];
			}
			return ret;
		}

		Vector3 mean(const vectorfield & vf)
		{
			Vector3 ret = { 0,0,0 };
			for (unsigned int i = 0; i<vf.size(); ++i)
			{
				ret = (i-1)/i * ret + vf[i]/i;
			}
			return ret;
		}




        __global__ void cu_dot(const Vector3 *vf1, const Vector3 *vf2, double *out, size_t N)
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
            for (int i=0; i<n; ++i)
            {
                ret += sf[i];
            }
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

            // Dot product
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

            // Dot product
            cu_add_c_a2<<<(n+1023)/1024, 1024>>>(c, a.data(), out.data(), n);
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

            // Dot product
            cu_add_c_dot<<<(n+1023)/1024, 1024>>>(c, a, b.data(), out.data(), n);
            cudaDeviceSynchronize();
        }

        __global__ void cu_add_c_dot2(scalar c, const Vector3 * a, const Vector3 * b, scalar * out, size_t N)
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

            // Dot product
            cu_add_c_dot2<<<(n+1023)/1024, 1024>>>(c, a.data(), b.data(), out.data(), n);
            cudaDeviceSynchronize();
        }


        // out[i] += c * a x b[i]
        void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
        {
			for (unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a.cross(b[idx]);
			}
        }

        // out[i] += c * a[i] x b[i]
        void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
        {
			for (unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a[idx].cross(b[idx]);
			}
        }

    }
}

#endif