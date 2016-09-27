#pragma once
#ifndef UTILITY_VECTORMATH_H
#define UTILITY_VECTORMATH_H

#include <vector>


namespace Utility
{
	namespace Vectormath
	{
		// Returns the Bohr Magneton
		double MuB();
		double kB();

		//Prints a 1-d array of doubles to console
		void Array_to_Console(const double *array, const int length);
		//Prints a 1-d array of ints to console
		void Array_to_Console(const int *array, const int length);


		// Builds the spins and spin_pos array according to nTa, nTb, nTc, nos_basic
		void Build_Spins(double ** &spins, double ** &spin_pos, const double *a, const double *b, const double *c, const int nCa, const int nCb, const int nCc, const int nos_basic);
		void Build_Spins(std::vector<double> &spin_pos, std::vector<std::vector<double>> & basis_atoms, std::vector<std::vector<double>> &translation_vectors, std::vector<int> &n_cells, const int nos_basic);


		//Multiplies a given 1-d array with a double skalar
		void Array_Skalar_Mult(double *array, const int length, const double skalar);

		//Adds array s1 * b onto a (both length length)
		void Array_Array_Add(double *a, const double *b, const int length, const double s1);
		void Array_Array_Add(double *a, const std::vector<double> b, const int length, const double s1);
		void Array_Array_Add(std::vector<double> &a, const std::vector<double> b, const double s1);

		//Adds two arrays a and b and outputs into c
		void Array_Array_Add(const double *a, const double *b, double *c, const int length);

		//Adds two arrays a and b and outputs into c = a *skalar + b *skalar_2
		void Array_Array_Add(const double *a, const double *b, double *c, const int length, const double skalar, const double skalar_2);
		void Array_Array_Add(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, const double skalar, const double skalar_2);

		//Adds two arrays a and b and outputs into result = a *skalar + b *skalar_2 + c *skalar_3
		void Array_Array_Add(const double *a, const double *b, const double *c, double *result, const int length, const double skalar, const double skalar_2, const double skalar_3);
		void Array_Array_Add(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &c, std::vector<double> &result, const double skalar, const double skalar_2, const double skalar_3);

		//Returns the Kartesian Length of an 
		//arbitrary long vector represented by a 1d-array 
		template <typename T>
		T Length(const T *array, const int length);
		template <typename T>
		T Length(const std::vector<T> &vec);

		// Returns the length of spin ispin from spins array vec
		template <typename T>
		T Length_3Nos(const std::vector<T> &vec, int ispin);

		//Returns the kartesian length of an arbitrary long vector
		//in the first slot of a 2D-Array at position pos_1
		template <typename T>
		T Length(const T *const *array, const int length, const int pos_1);
		template <typename T>
		T Length(const std::vector<std::vector<T>> &vec, const int pos_1);

		//Returns the kartesian length of an arbitrary long vector
		//in the first slot of a 3D-Array at position pos_1, pos_2
		template <typename T>
		T Length(const T *const *const *array, const int length, const int pos_1, const int pos_2);
		template <typename T>
		T Length(const std::vector<std::vector<std::vector<T>>> &vec, const int pos_1, const int pos_2);

		//Returns the kartesian length of an arbitrary long vector
		//in the first slot of a 4D-Array at position pos_1, pos_2, pos_3
		template <typename T>
		T Length(const T *const *const *const *array, const int length, const int pos_1, const int pos_2, const int pos_3);

		//Calculates the dot product of two 3D vectors
		double Dot_Product(const double *v1, const double *v2);
		double Dot_Product(const std::vector<double> &v1, const std::vector<double> &v2);
		double Dot_Product(const double *v1, const double * const* v2, const int b1);
		double Dot_Product(const double * const* v1, const int a1, const double * const* v2, const int b1);
		double Dot_Product(const double * const* v1, const int a1, const double *v2);
		double Dot_Product(const double * const* const* v1, const int a1, const int a2, const double *v2);

		//Calculates the Cross product of v1 and v2 and write it into v3
		void Cross_Product(const double *v1, const double *v2, double *v3);
		void Cross_Product(const double *v1, const std::vector<double> & v2, double *v3);
		void Cross_Product(const std::vector<double> &v1, const std::vector<double> &v2, std::vector<double> &v3);
		void Cross_Product(const double *const* v1, const int a1, const double *const* v2, const int b1, double *v3);
		void Cross_Product(const double *const* v1, const int a1, const double *const* const* v2, const int b1, const int b2, double *v3);

		//Normalizes an arbitrary vector of doubles with its Cartesian Length
		template <typename T>
		void Normalize(T *array, const int length);
		void Normalize(std::vector<double> &vec);
		// Normalize each 3-component subvector of a 3N-dimensional vector
		void Normalize_3Nos(std::vector<double> &vec);

		//Normalizes an array of array[dim][no_v] of vectors with their respective Kartesian Length
		void Normalize(const int dim, const int no_v, double **array);

		//Calculates the angle between two 3D vectors
		// degrees for unit==true; radians for unit==false
		double Angle(const double *v1, const double *v2, bool unit);
		double Angle(const std::vector<double> &v1, const std::vector<double> &v2, bool unit);

		//====================================== VECTOR COPY =======================================
		//Copies Vector b of dimension dim_w into a
		template <typename T1> void Vector_Copy(T1 * a, const T1 *b, const int dim);
		template <typename T1> void Vector_Copy(T1 * a, const T1 *const *b, const int dim, const int arg, const int pos_x);
		template <typename T1> void Vector_Copy(T1 * a, const T1 *const *const *b, const int dim, const int arg, const int pos_x, const int pos_y);
		template <typename T1> void Vector_Copy(T1 * a, const T1 *const *const *const *b, const int dim, const int arg, const int pos_x, const int pos_y, const int pos_z);

		template <typename T1> void Vector_Copy_io(std::vector<T1> & a, const std::vector<T1> & b);
		template <typename T1> void Vector_Copy_o(std::vector<T1> & a, const std::vector<std::vector<T1>> & b, const int arg, const int pos_x);
		template <typename T1> void Vector_Copy_o(std::vector<T1> & a, const std::vector<std::vector<std::vector<T1>>> &b, const int arg, const int pos_x, const int pos_y);
		template <typename T1> void Vector_Copy_o(std::vector<T1> & a, const std::vector<std::vector<std::vector<std::vector<T1>>>> &b, const int arg, const int pos_x, const int pos_y, const int pos_z);

		template <typename T1> void Vector_Copy(T1 ** a, const T1 *b, const int dim, const int arg, const int pos_x);
		template <typename T1> void Vector_Copy(T1 *** a, const T1 *b, const int dim, const int arg, const int pos_x, const int pos_y);
		template <typename T1> void Vector_Copy(T1 **** a, const T1 *b, const int dim, const int arg, const int pos_x, const int pos_y, const int pos_z);

		template <typename T1> void Vector_Copy_i(std::vector<std::vector<T1>> & a, const std::vector<T1> &b, const int arg, const int pos_x);
		template <typename T1> void Vector_Copy_i(std::vector<std::vector<std::vector<T1>>> &a, const std::vector<T1> &b, const int arg, const int pos_x, const int pos_y);
		template <typename T1> void Vector_Copy_i(std::vector<std::vector<std::vector<std::vector<T1>>>> &a, const std::vector<T1> &b, const int arg, const int pos_x, const int pos_y, const int pos_z);

		//====================================== Array  COPY =======================================
		// Copies Array b of dimensions dim_w, dim_x into a
		template <typename T1> void Array_Copy(T1 **a, const T1 *const *b, const int dim_w, const int dim_x);

		//============================= ALLOCATING RECTANGULAR ARRAYS ==============================
		template <typename T1> void Alloc_Array(T1 * &a, const int dim_w);
		template <typename T1> void Alloc_Array(T1 ** &a, const int dim_w, const int dim_x);
		template <typename T1> void Alloc_Array(T1 *** &a, const int dim_w, const int dim_x, const int dim_y);
		template <typename T1> void Alloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int dim_y, const int dim_z);

		//============================= Copy RECTANGULAR ARRAYS ==============================
		template <typename T1> void Copy_Array(T1 * &a, T1 *b, const int dim_w);
		template <typename T1> void Copy_Array(T1 ** &a, T1 **b, const int dim_w, const int dim_x);
		template <typename T1> void Copy_Array(T1 *** &a, T1 ***b, const int dim_w, const int dim_x, const int dim_y);
		template <typename T1> void Copy_Array(T1 **** &a, T1 ****b, const int dim_w, const int dim_x, const int dim_y, const int dim_z);

		//============================ DEALLOCATING RECTANGULAR ARRAYS =============================
		template <typename T1> void Dealloc_Array(T1 * &a);
		template <typename T1> void Dealloc_Array(T1 ** &a, const int dim_w);
		template <typename T1> void Dealloc_Array(T1 *** &a, const int dim_w, const int dim_x);
		template <typename T1> void Dealloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int dim_y);


		//============================== ALLOCATING DYNAMIC ARRAYS =================================
		// 3D array dyn with dim_y[dim_x]
		template <typename T1> void Alloc_Array(T1 *** &a, const int dim_w, const int dim_x, const int *const &dim_y);
		template <typename T1> void Copy_Array(T1 *** &a, T1 *** b, const int dim_w, const int dim_x, const int *const &dim_y);
		// 4D array dyn with dim_z[dim_y]
		template <typename T1> void Alloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int dim_y, const int *const &dim_z);
		template <typename T1> void Copy_Array(T1 **** &a, T1 ****b, const int dim_w, const int dim_x, const int dim_y, const int *const &dim_z);
		//============================= DEALLOCATING DYNAMIC ARRAYS ================================
		// 3D array dyn with dim_y[dim_x] per normal dealloc
		// 4D array dyn with dim_y[dim_x]
		template <typename T1> void Dealloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int *const &dim_y);
	}
}
#include "Vectormath.hpp"
#endif