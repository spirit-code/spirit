#ifndef USE_CUDA

#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <algorithm>

namespace Engine
{
	namespace Vectormath
	{

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


		void Build_Spins(vectorfield & spin_pos, const std::vector<Vector3> & basis_atoms, const std::vector<Vector3> & translation_vectors, const intfield & n_cells)
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
									+ ka*translation_vectors[0] + k2*translation_vectors[1] + k3*translation_vectors[2]);
								if ((i != j || ka != 0 || k2 != 0 || k3 != 0) && std::abs(sp[0]) < 1e-9 && std::abs(sp[1]) < 1e-9 && std::abs(sp[2]) < 1e-9)
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
			int i, j, k, s, ispin;
			int nos_basic = basis_atoms.size();
			//int nos = nos_basic * n_cells[0] * n_cells[1] * n_cells[2];
			Vector3 build_array;
			for (k = 0; k < n_cells[2]; ++k) {
				for (j = 0; j < n_cells[1]; ++j) {
					for (i = 0; i < n_cells[0]; ++i) {
						for (s = 0; s < nos_basic; ++s) {
							ispin = k*n_cells[1] * n_cells[0] * nos_basic + j*n_cells[0] * nos_basic + i*nos_basic + s;
							build_array = i*translation_vectors[0] + j*translation_vectors[1] + k*translation_vectors[2];
							// paste initial spin orientations across the lattice translations
							//spins[dim*nos + ispin] = spins[dim*nos + s];
							// calculate the spin positions
							spin_pos[ispin] = basis_atoms[s] + build_array;
						}// endfor s
					}// endfor k
				}// endfor j
			}// endfor dim

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

		scalar TopologicalCharge(const vectorfield & vf)
		{
        	Log(Utility::Log_Level::Warning, Utility::Log_Sender::All, std::string("Calculating the topological charge is not yet implemented"));
			return 0;
		}

		// Utility function for the SIB Optimizer
		void transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
		{
			Vector3 e1, a2, A;
			scalar detAi;
			for (unsigned int i = 0; i < spins.size(); ++i)
			{
				e1 = spins[i];
				A = force[i];

				// 1/determinant(A)
				detAi = 1.0 / (1 + pow(A.norm(), 2.0));

				// calculate equation without the predictor?
				a2 = e1 + e1.cross(A);

				out[i][0] = (a2[0] * (1 + A[0] * A[0])    + a2[1] * (A[0] * A[1] + A[2]) + a2[2] * (A[0] * A[2] - A[1]))*detAi;
				out[i][1] = (a2[0] * (A[1] * A[0] - A[2]) + a2[1] * (1 + A[1] * A[1])    + a2[2] * (A[1] * A[2] + A[0]))*detAi;
				out[i][2] = (a2[0] * (A[2] * A[0] + A[1]) + a2[1] * (A[2] * A[1] - A[0]) + a2[2] * (1 + A[2] * A[2]))*detAi;
			}
		}
		void get_random_vectorfield(const Data::Spin_System & sys, scalar epsilon, vectorfield & xi)
		{
			for (int i = 0; i < sys.nos; ++i)
			{
				for (int dim = 0; dim < 3; ++dim)
				{
					// PRNG gives RN int [0,1] -> [-1,1] -> multiply with epsilon
					xi[i][dim] = epsilon*(sys.llg_parameters->distribution_int(sys.llg_parameters->prng) * 2 - 1);
				}
			}
		}



		/////////////////////////////////////////////////////////////////

		void assign(scalarfield & sf_dest, const scalarfield& sf_source)
		{
			sf_dest = sf_source;
		}

		void fill(scalarfield & sf, scalar s)
		{
			for (unsigned int i = 0; i<sf.size(); ++i)
			{
				sf[i] = s;
			}
		}
		void fill(scalarfield & sf, scalar s, const intfield & mask)
		{
			for (unsigned int i=0; i<sf.size(); ++i)
			{
				sf[i] = mask[i]*s;
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
			scalar ret = sf[0];
			for (unsigned int i = 1; i<sf.size(); ++i)
			{
				ret += (sf[i] - ret) / i;
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
		void fill(vectorfield & vf, const Vector3 & v, const intfield & mask)
		{
			for (unsigned int i=0; i<vf.size(); ++i)
			{
				vf[i] = mask[i]*v;
			}
		}

		void normalize_vectors(vectorfield & vf)
		{
			for (unsigned int i=0; i<vf.size(); ++i)
			{
				vf[i].normalize();
			}
		}
		
		std::pair<scalar, scalar> minmax_component(const vectorfield & v1)
		{
			scalar min=1e6, max=-1e6;
			std::pair<scalar, scalar> minmax;
			for (unsigned int i = 0; i < v1.size(); ++i)
			{
				for (int dim = 0; dim < 3; ++dim)
				{
					if (v1[i][dim] < min) min = v1[i][dim];
					if (v1[i][dim] > max) max = v1[i][dim];
				}
			}
			minmax.first = min;
			minmax.second = max;
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
			for (unsigned int i = 1; i<vf.size(); ++i)
			{
				ret += (vf[i] - ret) / i;
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

		// out[i] = c*a
		void set_c_a(const scalar & c, const Vector3 & a, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = c*a;
			}
		}
		// out[i] = c*a
		void set_c_a(const scalar & c, const Vector3 & a, vectorfield & out, const intfield & mask)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = mask[idx]*c*a;
			}
		}

		// out[i] = c*a[i]
		void set_c_a(const scalar & c, const vectorfield & a, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = c*a[idx];
			}
		}// out[i] = c*a[i]
		void set_c_a(const scalar & c, const vectorfield & a, vectorfield & out, const intfield & mask)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = mask[idx] * c*a[idx];
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

		// out[i] = c * a*b[i]
		void set_c_dot(const scalar & c, const Vector3 & a, const vectorfield & b, scalarfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = c*a.dot(b[idx]);
			}
		}
		// out[i] = c * a[i]*b[i]
		void set_c_dot(const scalar & c, const vectorfield & a, const vectorfield & b, scalarfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = c*a[idx].dot(b[idx]);
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
		
		// out[i] = c * a x b[i]
		void set_c_cross(const scalar & c, const Vector3 & a, const vectorfield & b, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = c*a.cross(b[idx]);
			}
		}
		// out[i] = c * a[i] x b[i]
		void set_c_cross(const scalar & c, const vectorfield & a, const vectorfield & b, vectorfield & out)
		{
			for(unsigned int idx = 0; idx < out.size(); ++idx)
			{
				out[idx] = c*a[idx].cross(b[idx]);
			}
		}
	}
}

#endif