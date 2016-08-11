#define _USE_MATH_DEFINES
#include <cmath>
#include "Vectormath.h"

#include "Logging.h"
//extern Utility::LoggingHandler Log;
#include "Exception.h"

#include"Logging.h"
//extern Utility::LoggingHandler Log;

namespace Utility
{
	namespace Vectormath
	{
		// Returns the Bohr Magneton [meV / T]
		double MuB()
		{
			return 0.057883817555;
		}
		// Returns the Boltzmann constant [meV / K]
		double kB()
		{
			return 0.08617330350;
		}
		///////////// Quick Debug Method that is used occasionally to print some arrays -> move to Utility_IO?!
		void Array_to_Console(const double *array, const int length) {
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
		void Build_Spins(double ** &spins, double ** &spin_pos, const double *a, const double *b, const double *c, const int nCa, const int nCb, const int nCc, const int nos_basic)
		{
			int i, j, k, s, pos, dim;
			double build_array[3] = { 0 };
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
								Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, "Unable to initialize Spin-System, since 2 spins occupy the same space.\nPlease check the config file!");
								throw Exception::System_not_Initialized;
							}
						}
					}
				}
			}
		};// end Build_Spins

		void Build_Spins(std::vector<std::vector<double>> &spin_pos, std::vector<std::vector<double>> & basis_atoms, std::vector<std::vector<double>> &translation_vectors, std::vector<int> &n_cells, const int nos_basic)
		{
			double a[3] = { translation_vectors[0][0], translation_vectors[1][0], translation_vectors[2][0] };
			double b[3] = { translation_vectors[0][1], translation_vectors[1][1], translation_vectors[2][1] };
			double c[3] = { translation_vectors[0][2], translation_vectors[1][2], translation_vectors[2][2] };

			int i, j, k, s, pos, dim, nos = nos_basic * n_cells[0] * n_cells[1] * n_cells[2];
			double build_array[3] = { 0 };
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
								spin_pos[dim][pos] = basis_atoms[dim][s] + build_array[dim];
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
								Log.Send(Utility::Log_Level::SEVERE, Utility::Log_Sender::ALL, "Unable to initialize Spin-System, since 2 spins occupy the same space.\nPlease check the config file!");
								throw Exception::System_not_Initialized;
							}
						}
					}
				}
			}
		};// end Build_Spins


		void Array_Skalar_Mult(double *array, const int length, const double skalar) {
			for (int i = 0; i < length; ++i) {
				array[i] = array[i] * skalar;
			}
		};//end Array_Skalar_Mult
		void Array_Array_Add(double * a, const double * b, const int length, const double s1)
		{
			for (int i = 0; i < length; ++i) {
				a[i] = a[i] + s1*b[i];
			}
		}
		void Array_Array_Add(double * a, const std::vector<double> b, const int length, const double s1)
		{
			for (int i = 0; i < length; ++i) {
				a[i] = a[i] + s1*b[i];
			}
		}
		void Array_Array_Add(std::vector<double>& a, const std::vector<double> b, const double s1)
		{
			for (int i = 0; i < (int)a.size(); ++i) {
				a[i] = a[i] + s1*b[i];
			}
		}
		void Array_Array_Add(const double *a, const double *b, double *c, const int length) {
			for (int i = 0; i < length; ++i) {
				c[i] = a[i] + b[i];
			}
		};
		void Array_Array_Add(const double *a, const double *b, double *c, const int length, const double skalar, const double skalar_2) {
			for (int i = 0; i < length; ++i) {
				c[i] = skalar*a[i] + skalar_2*b[i];
			}
		}
		void Array_Array_Add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, const double skalar, const double skalar_2)
		{
			for (int i = 0; i < (int)a.size(); ++i) {
				c[i] = skalar*a[i] + skalar_2*b[i];
			}
		}
		;
		void Array_Array_Add(const double *a, const double *b, const double *c, double *result, const int length, const double skalar, const double skalar_2, const double skalar_3) {
			for (int i = 0; i < length; ++i) {
				result[i] = skalar*a[i] + skalar_2*b[i] + skalar_3*c[i];
			}
		}
		void Array_Array_Add(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, std::vector<double>& result, const double skalar, const double skalar_2, const double skalar_3)
		{
			for (int i = 0; i < (int)a.size(); ++i) {
				result[i] = skalar*a[i] + skalar_2*b[i] + skalar_3*c[i];
			}
		}
		;//end Array_Array_Add

		double Length(const double *array, const int length) {
			double result = 0;
			for (int i = 0; i < length; ++i) {
				result += pow(array[i], 2.0);
			}
			return sqrt(result);
		}
		double Length(const std::vector<double> &vec)
		{
			double result = 0.0;
			for (int i = 0; i < (int)vec.size(); ++i) {
				result += pow(vec[i], 2.0);
			}
			return sqrt(result);
		}
		double Length_3Nos(const std::vector<double>& vec, int ispin)
		{
			double result = 0.0;
			for (int dim = 0; dim < 3; ++dim) {
				result += pow(vec[dim * (int)vec.size()/3 + ispin], 2.0);
			}
			return sqrt(result);
		}
		;
		double Length(const double *const *array, const int length, const int pos_1) {
			double result = 0;
			for (int i = 0; i < length; ++i) {
				result += pow(array[i][pos_1], 2.0);
			}
			return sqrt(result);
		}
		double Length(const std::vector<std::vector<double>>& vec, const int pos_1)
		{
			double result = 0;
			for (int i = 0; i < (int)vec.size(); ++i) {
				result += pow(vec[i][pos_1], 2.0);
			}
			return sqrt(result);
		}
		;
		double Length(const double *const *const *array, const int length, const int pos_1, const int pos_2) {
			double result = 0;
			for (int i = 0; i < length; ++i) {
				result += pow(array[i][pos_1][pos_2], 2.0);
			}
			return sqrt(result);
		}
		double Length(const std::vector<std::vector<std::vector<double>>>& vec, const int pos_1, const int pos_2)
		{
			double result = 0;
			for (int i = 0; i < (int)vec.size(); ++i) {
				result += pow(vec[i][pos_1][pos_2], 2.0);
			}
			return sqrt(result);
		}
		;
		double Length(const double *const *const *const *array, const int length, const int pos_1, const int pos_2, const int pos_3) {
			double result = 0;
			for (int i = 0; i < length; ++i) {
				result += pow(array[i][pos_1][pos_2][pos_3], 2.0);
			}
			return sqrt(result);
		};//end Length

		void Normalize(double *array, const int length) {
			double l = Length(array, length);
			if (l == 0.0) { throw Exception::Division_by_zero; };
			double norm = 1.0 / l;
			for (int i = 0; i < length; ++i) {
				array[i] = array[i] * norm;
			}
		}
		void Normalize(std::vector<double> &vec)
		{
			double l = Length(vec);
			if (l == 0.0) { throw Exception::Division_by_zero; };
			double norm = 1.0 / l;
			for (int i = 0; i < (int)vec.size(); ++i) {
				vec[i] = vec[i] * norm;
			}
		}
		void Normalize_3Nos(std::vector<double>& vec)
		{
			double l = 0.0;
			int nos = vec.size() / 3;
			for (int iatom = 0; iatom < nos; ++iatom) {
				l = Length_3Nos(vec, iatom);
				if (l == 0.0) { throw Exception::Division_by_zero; };
				double norm = 1.0 / l;
				for (int dim = 0; dim < 3; ++dim) {
					vec[dim * nos + iatom] = vec[dim * nos + iatom] * norm;
				}
			}
		}
		//Normalizes an array of array[dim][no_v] of vectors with their respective Kartesian Length
		void Normalize(const int dim, const int no_v, double ** array)
		{
			double norm = 0, l = 0;
			for (int i = 0; i < no_v; ++i) {
				double l = Length(array, dim, i);
				if (l == 0.0) { throw Exception::Division_by_zero; };
				norm = 1.0 / l;
				for (int j = 0; j < dim; ++j) {
					array[j][i] = array[j][i] * norm;
				}
			}
		}
		;//end Normalize

		//Calculates the dot product of two 3D vectors
		double Dot_Product(const double *v1, const double *v2) {
			return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
		}
		double Dot_Product(const std::vector<double>& v1, const std::vector<double>& v2)
		{
			return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
		}
		double Dot_Product(const double * v1, const double * const* v2, const int b1)
		{
			return v1[0] * v2[0][b1] + v1[1] * v2[1][b1] + v1[2] * v2[2][b1];
		}
		double Dot_Product(const double *const* v1, const int a1, const double *const* v2, const int b1)
		{
			return v1[0][a1] * v2[0][b1] + v1[1][a1] * v2[1][b1] + v1[2][a1] * v2[2][b1];
		}
		double Dot_Product(const double * const * v1, const int a1, const double * v2)
		{
			return v1[0][a1] * v2[0] + v1[1][a1] * v2[1] + v1[2][a1] * v2[2];
		}
		double Dot_Product(const double * const * const * v1, const int a1, const int a2, const double * v2)
		{
			return v1[0][a1][a2] * v2[0] + v1[1][a1][a2] * v2[1] + v1[2][a1][a2] * v2[2];
		}
		//end Dot_Product

		 //Calculates the Cross product of v1 and v2 and write it into v3
		void Cross_Product(const double *v1, const double *v2, double *v3) {
			v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		void Cross_Product(const double * v1, const std::vector<double>  &v2, double * v3)
		{
			v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		void Cross_Product(const std::vector<double>& v1, const std::vector<double>& v2, std::vector<double>& v3)
		{
			v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
			v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
			v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
		}
		void Cross_Product(const double * const * v1, const int a1, const double * const * v2, const int b1, double * v3)
		{
			v3[0] = v1[1][a1] * v2[2][b1] - v1[2][a1] * v2[1][b1];
			v3[1] = v1[2][a1] * v2[0][b1] - v1[0][a1] * v2[2][b1];
			v3[2] = v1[0][a1] * v2[1][b1] - v1[1][a1] * v2[0][b1];
		}
		void Cross_Product(const double * const * v1, const int a1, const double * const * const * v2, const int b1, const int b2, double * v3)
		{
			v3[0] = v1[1][a1] * v2[2][b1][b2] - v1[2][a1] * v2[1][b1][b2];
			v3[1] = v1[2][a1] * v2[0][b1][b2] - v1[0][a1] * v2[2][b1][b2];
			v3[2] = v1[0][a1] * v2[1][b1][b2] - v1[1][a1] * v2[0][b1][b2];
		}

		//Calculates the angle between two 3D vectors
		// degrees for unit==true; radians for unit==false
		double Angle(const double *v1, const double *v2, bool unit) {
			double angle = Dot_Product(v1, v2) / Length(v1, 3) / Length(v2, 3);
			angle = std::acos(angle);
			if (unit) {
				angle = angle * 180.0 / M_PI;
			}
			return angle;
		}
		double Angle(const std::vector<double>& v1, const std::vector<double>& v2, bool unit)
		{
			double angle = Dot_Product(v1, v2) / Length(v1) / Length(v2);
			angle = std::acos(angle);
			if (unit) {
				angle = angle * 180.0 / M_PI;
			}
			return angle;
		}
		;//end Angle
	}
}