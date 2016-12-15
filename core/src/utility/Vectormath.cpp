#define _USE_MATH_DEFINES
#include <cmath>
#include "Vectormath.hpp"

#include "Logging.hpp"
//extern Utility::LoggingHandler Log;
#include "Exception.hpp"

#include"Logging.hpp"
//extern Utility::LoggingHandler Log;

namespace Utility
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
		///////////// Quick Debug Method that is used occasionally to print some arrays -> move to Utility_IO?!
		void Array_to_Console(const scalar *array, const int length) {
			std::cout << std::endl;
			for (int i = 0; i < length; ++i) {
				std::cout << array[i] << ' ';
			}
			std::cout << std::endl;
		};//end Array_to_Console
		void Array_to_Console(const int *array, const int length) {
			std::cout << std::endl;
			for (int i = 0; i < length; ++i) {
				std::cout << array[i] << ' ';
			}
			std::cout << std::endl;
		};//end Array_to_Console
	/////////////////////

		/*
			Build_Spins creates the orientation and position vectors for all spins from shape, basis and translations (nCells) information
		*/
		void Build_Spins(scalar ** &spins, scalar ** &spin_pos, const scalar *a, const scalar *b, const scalar *c, const int nCa, const int nCb, const int nCc, const int nos_basic)
		{
			int i, j, k, s, pos, dim;
			scalar build_array[3] = { 0 };
			for (k = 0; k < nCc; ++k) {
				for (j = 0; j < nCb; ++j) {
					for (i = 0; i < nCa; ++i) {
						for (s = 0; s < nos_basic; ++s) {
							pos = k*nCb*nCa*nos_basic + j*nCa*nos_basic + i*nos_basic + s;
							Vectormath::Array_Array_Add(a, b, c, build_array, 3, i, j, k);
							for (dim = 0; dim < 3; ++dim)
							{
								// paste initial spin orientations across the lattice translations
								spins[dim][pos] = spins[dim][s];
								// calculate the spin positions
								spin_pos[dim][pos] = spin_pos[dim][s] + build_array[dim];
							}// endfor dim
						}// endfor s
					}// endfor k
				}// endfor j
			}// endfor dim

			// Check for erronous input placing two spins on the same location
			for (i = 0; i < pos; ++i) {
				for (j = i + 1; j < pos; ++j) {
					if (std::abs(spin_pos[0][i] - spin_pos[0][j]) < 1.0E-6) {
						if (std::abs(spin_pos[1][i] - spin_pos[1][j]) < 1.0E-6) {
							if (std::abs(spin_pos[2][i] - spin_pos[2][j]) < 1.0E-6) {
								Log(Utility::Log_Level::Severe, Utility::Log_Sender::All, "Unable to initialize Spin-System, since 2 spins occupy the same space.\nPlease check the config file!");
								throw Exception::System_not_Initialized;
							}
						}
					}
				}
			}
		};// end Build_Spins

		void Build_Spins(std::vector<scalar> &spin_pos, std::vector<std::vector<scalar>> & basis_atoms, std::vector<std::vector<scalar>> &translation_vectors, std::vector<int> &n_cells, const int nos_basic)
		{
			scalar a[3] = { translation_vectors[0][0], translation_vectors[1][0], translation_vectors[2][0] };
			scalar b[3] = { translation_vectors[0][1], translation_vectors[1][1], translation_vectors[2][1] };
			scalar c[3] = { translation_vectors[0][2], translation_vectors[1][2], translation_vectors[2][2] };

			int i, j, k, s, pos, dim, nos = nos_basic * n_cells[0] * n_cells[1] * n_cells[2];
			scalar build_array[3] = { 0 };
			for (k = 0; k < n_cells[2]; ++k) {
				for (j = 0; j < n_cells[1]; ++j) {
					for (i = 0; i < n_cells[0]; ++i) {
						for (s = 0; s < nos_basic; ++s) {
							pos = k*n_cells[1]*n_cells[0]*nos_basic + j*n_cells[0]*nos_basic + i*nos_basic + s;
							Vectormath::Array_Array_Add(a, b, c, build_array, 3, i, j, k);
							for (dim = 0; dim < 3; ++dim)
							{
								// paste initial spin orientations across the lattice translations
								//spins[dim*nos + pos] = spins[dim*nos + s];
								// calculate the spin positions
								spin_pos[dim*nos+pos] = basis_atoms[dim][s] + build_array[dim];
							}// endfor dim
						}// endfor s
					}// endfor k
				}// endfor j
			}// endfor dim

			// Check for erronous input placing two spins on the same location
			std::vector<scalar> sp(3);
			for (unsigned int i = 0; i < basis_atoms[0].size(); ++i)
			{
				for (unsigned int j = 0; j < basis_atoms[0].size(); ++j)
				{
					for (int k1 = -2; k1 < 3; ++k1)
					{
						for (int k2 = -2; k2 < 3; ++k2)
						{
							for (int k3 = -2; k3 < 3; ++k3)
							{
								// Norm is zero if translated basis atom is at position of another basis atom
								for (int dim = 0; dim < 3; ++dim)
								{
									sp[dim] = basis_atoms[dim][i] - ( basis_atoms[dim][j]
										+ k1*translation_vectors[dim][0] + k2*translation_vectors[dim][1] + k3*translation_vectors[dim][2] );
								}
								if ( (i!=j || k1!=0 || k2!=0 || k3!=0) && std::abs(sp[0]) < 1e-9 && std::abs(sp[1]) < 1e-9 && std::abs(sp[2]) < 1e-9)
								{
									Log(Utility::Log_Level::Severe, Utility::Log_Sender::All, "Unable to initialize Spin-System, since 2 spins occupy the same space.\nPlease check the config file!");
									Log.Append_to_File();
									throw Exception::System_not_Initialized;
								}
							}
						}
					}
				}
			}
		};// end Build_Spins


		void Array_Skalar_Mult(scalar *array, const int length, const scalar skalar) {
			for (int i = 0; i < length; ++i) {
				array[i] = array[i] * skalar;
			}
		};//end Array_Skalar_Mult
		void Array_Array_Add(scalar * a, const scalar * b, const int length, const scalar s1)
		{
			for (int i = 0; i < length; ++i) {
				a[i] = a[i] + s1*b[i];
			}
		}
		void Array_Array_Add(scalar * a, const std::vector<scalar> b, const int length, const scalar s1)
		{
			for (int i = 0; i < length; ++i) {
				a[i] = a[i] + s1*b[i];
			}
		}
		void Array_Array_Add(std::vector<scalar>& a, const std::vector<scalar> b, const scalar s1)
		{
			for (int i = 0; i < (int)a.size(); ++i) {
				a[i] = a[i] + s1*b[i];
			}
		}
		void Array_Array_Add(const scalar *a, const scalar *b, scalar *c, const int length) {
			for (int i = 0; i < length; ++i) {
				c[i] = a[i] + b[i];
			}
		};
		void Array_Array_Add(const scalar *a, const scalar *b, scalar *c, const int length, const scalar skalar, const scalar skalar_2) {
			for (int i = 0; i < length; ++i) {
				c[i] = skalar*a[i] + skalar_2*b[i];
			}
		}
		void Array_Array_Add(const std::vector<scalar>& a, const std::vector<scalar>& b, std::vector<scalar>& c, const scalar skalar, const scalar skalar_2)
		{
			for (int i = 0; i < (int)a.size(); ++i) {
				c[i] = skalar*a[i] + skalar_2*b[i];
			}
		}
		;
		void Array_Array_Add(const scalar *a, const scalar *b, const scalar *c, scalar *result, const int length, const scalar skalar, const scalar skalar_2, const scalar skalar_3) {
			for (int i = 0; i < length; ++i) {
				result[i] = skalar*a[i] + skalar_2*b[i] + skalar_3*c[i];
			}
		}
		void Array_Array_Add(const std::vector<scalar>& a, const std::vector<scalar>& b, const std::vector<scalar>& c, std::vector<scalar>& result, const scalar skalar, const scalar skalar_2, const scalar skalar_3)
		{
			for (int i = 0; i < (int)a.size(); ++i) {
				result[i] = skalar*a[i] + skalar_2*b[i] + skalar_3*c[i];
			}
		};//end Array_Array_Add

		
		void Normalize(std::vector<scalar> &vec)
		{
			scalar l = Length(vec);
			if (l == 0.0) { throw Exception::Division_by_zero; };
			scalar norm = 1.0 / l;
			for (int i = 0; i < (int)vec.size(); ++i) {
				vec[i] = vec[i] * norm;
			}
		}
		void Normalize_3Nos(std::vector<scalar>& vec)
		{
			scalar l = 0.0;
			int nos = vec.size() / 3;
			for (int iatom = 0; iatom < nos; ++iatom) {
				l = Length_3Nos(vec, iatom);
				if (l == 0.0) { throw Exception::Division_by_zero; };
				scalar norm = 1.0 / l;
				for (int dim = 0; dim < 3; ++dim) {
					vec[dim * nos + iatom] = vec[dim * nos + iatom] * norm;
				}
			}
		}
		//Normalizes an array of array[dim][no_v] of vectors with their respective Kartesian Length
		void Normalize(const int dim, const int no_v, scalar ** array)
		{
			scalar norm = 0, l = 0;
			for (int i = 0; i < no_v; ++i) {
				scalar l = Length(array, dim, i);
				if (l == 0.0) { throw Exception::Division_by_zero; };
				norm = 1.0 / l;
				for (int j = 0; j < dim; ++j) {
					array[j][i] = array[j][i] * norm;
				}
			}
		}
		;//end Normalize

		//Calculates the dot product of two 3D vectors
		scalar Dot_Product(const scalar *v1, const scalar *v2) {
			return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
		}
		scalar Dot_Product(const std::vector<scalar>& v1, const std::vector<scalar>& v2)
		{
			return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
		}
		scalar Dot_Product(const scalar * v1, const scalar * const* v2, const int b1)
		{
			return v1[0] * v2[0][b1] + v1[1] * v2[1][b1] + v1[2] * v2[2][b1];
		}
		scalar Dot_Product(const scalar *const* v1, const int a1, const scalar *const* v2, const int b1)
		{
			return v1[0][a1] * v2[0][b1] + v1[1][a1] * v2[1][b1] + v1[2][a1] * v2[2][b1];
		}
		scalar Dot_Product(const scalar * const * v1, const int a1, const scalar * v2)
		{
			return v1[0][a1] * v2[0] + v1[1][a1] * v2[1] + v1[2][a1] * v2[2];
		}
		scalar Dot_Product(const scalar * const * const * v1, const int a1, const int a2, const scalar * v2)
		{
			return v1[0][a1][a2] * v2[0] + v1[1][a1][a2] * v2[1] + v1[2][a1][a2] * v2[2];
		}
		//end Dot_Product

		 //Calculates the Cross product of v1 and v2 and write it into v3
		void Cross_Product(const scalar *v1, const scalar *v2, scalar *v3) {
			v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		void Cross_Product(const scalar * v1, const std::vector<scalar>  &v2, scalar * v3)
		{
			v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		void Cross_Product(const std::vector<scalar>& v1, const std::vector<scalar>& v2, std::vector<scalar>& v3)
		{
			v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		void Cross_Product(const scalar * const * v1, const int a1, const scalar * const * v2, const int b1, scalar * v3)
		{
			v3[0] = v1[1][a1] * v2[2][b1] - v1[2][a1] * v2[1][b1];
			v3[1] = v1[2][a1] * v2[0][b1] - v1[0][a1] * v2[2][b1];
			v3[2] = v1[0][a1] * v2[1][b1] - v1[1][a1] * v2[0][b1];
		}
		void Cross_Product(const scalar * const * v1, const int a1, const scalar * const * const * v2, const int b1, const int b2, scalar * v3)
		{
			v3[0] = v1[1][a1] * v2[2][b1][b2] - v1[2][a1] * v2[1][b1][b2];
			v3[1] = v1[2][a1] * v2[0][b1][b2] - v1[0][a1] * v2[2][b1][b2];
			v3[2] = v1[0][a1] * v2[1][b1][b2] - v1[1][a1] * v2[0][b1][b2];
		}

		//Calculates the angle between two 3D vectors
		// degrees for unit==true; radians for unit==false
		scalar Angle(const scalar *v1, const scalar *v2, bool unit) {
			scalar angle = Dot_Product(v1, v2) / Length(v1, 3) / Length(v2, 3);
			angle = std::acos(angle);
			if (unit) {
				angle = angle * 180.0 / M_PI;
			}
			return angle;
		}
		scalar Angle(const std::vector<scalar>& v1, const std::vector<scalar>& v2, bool unit)
		{
			scalar angle = Dot_Product(v1, v2) / Length(v1) / Length(v2);
			angle = std::acos(angle);
			if (unit) {
				angle = angle * 180.0 / M_PI;
			}
			return angle;
		}
		;//end Angle
	}
}