#ifndef USE_CUDA

#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>

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


		std::array<scalar,3> Magnetization(const vectorfield & vf)
		{
			std::array<scalar, 3> M{0, 0, 0};
			int nos = vf.size();
			scalar scale = 1/(scalar)nos;
			for (int i=0; i<nos; ++i)
			{
				M[0] += vf[i][0]*scale;
				M[1] += vf[i][1]*scale;
				M[2] += vf[i][2]*scale;
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

		void fill(vectorfield & vf, const Vector3 & v)
		{
			for (unsigned int i=0; i<vf.size(); ++i)
			{
				vf[i] = v;
			}
		}

		void scale(vectorfield & vf, const scalar & sc)
		{
			for (unsigned int i=0; i<vf.size(); ++i)
			{
				vf[i] *= sc;
			}
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


		// computes the inner product of two vectorfields v1 and v2
		scalar dot(const vectorfield & v1, const vectorfield & v2)
		{
			scalar x = 0;
			for (unsigned int i = 0; i<v1.size(); ++i)
			{
				x += v1[i].dot(v2[i]);
			}
			return x;
		}

		// computes the inner products of vectors in v1 and v2
		// v1 and v2 are vectorfields
		void dot(const vectorfield & v1, const vectorfield & v2, scalarfield & out)
		{
			for (unsigned int i=0; i<v1.size(); ++i)
			{
				out[i] = v1[i].dot(v2[i]);
			}
		}

		// computes the vector (cross) products of vectors in v1 and v2
		// v1 and v2 are vector fields
		void cross(const vectorfield & v1, const vectorfield & v2, vectorfield & out)
		{
			for (unsigned int i=0; i<v1.size(); ++i)
			{
				out[i] = v1[i].cross(v2[i]);
			}
		}


		// out[i] += c*a
		void add_c_a(const scalar & c, const Vector3 & a, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a;
			}
		}

		// out[i] += c*a[i]
		void add_c_a(const scalar & c, const vectorfield & a, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a[idx];
			}
		}


		// out[i] += c * a*b[i]
		void add_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a.dot(b[idx]);
			}
		}

		// out[i] += c * a[i]*b[i]
		void add_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a[idx].dot(b[idx]);
			}
		}


		// out[i] += c * a x b[i]
		void add_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a.cross(b[idx]);
			}
		}

		// out[i] += c * a[i] x b[i]
		void add_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] += c*a[idx].cross(b[idx]);
			}
		}
	}
}

#endif