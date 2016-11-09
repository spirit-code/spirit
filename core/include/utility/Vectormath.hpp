#pragma once
#ifndef UTILITY_VECTORMATH_H
#define UTILITY_VECTORMATH_H

#include <vector>

#include "Core_Defines.h"

namespace Utility
{
	namespace Vectormath
	{
		// Returns the Bohr Magneton
		scalar MuB();
		scalar kB();

		//Prints a 1-d array of scalars to console
		void Array_to_Console(const scalar *array, const int length);
		//Prints a 1-d array of ints to console
		void Array_to_Console(const int *array, const int length);


		// Builds the spins and spin_pos array according to nTa, nTb, nTc, nos_basic
		void Build_Spins(scalar ** &spins, scalar ** &spin_pos, const scalar *a, const scalar *b, const scalar *c, const int nCa, const int nCb, const int nCc, const int nos_basic);
		void Build_Spins(std::vector<scalar> &spin_pos, std::vector<std::vector<scalar>> & basis_atoms, std::vector<std::vector<scalar>> &translation_vectors, std::vector<int> &n_cells, const int nos_basic);


		//Multiplies a given 1-d array with a scalar skalar
		void Array_Skalar_Mult(scalar *array, const int length, const scalar skalar);

		//Adds array s1 * b onto a (both length length)
		void Array_Array_Add(scalar *a, const scalar *b, const int length, const scalar s1);
		void Array_Array_Add(scalar *a, const std::vector<scalar> b, const int length, const scalar s1);
		void Array_Array_Add(std::vector<scalar> &a, const std::vector<scalar> b, const scalar s1);

		//Adds two arrays a and b and outputs into c
		void Array_Array_Add(const scalar *a, const scalar *b, scalar *c, const int length);

		//Adds two arrays a and b and outputs into c = a *skalar + b *skalar_2
		void Array_Array_Add(const scalar *a, const scalar *b, scalar *c, const int length, const scalar skalar, const scalar skalar_2);
		void Array_Array_Add(const std::vector<scalar> &a, const std::vector<scalar> &b, std::vector<scalar> &c, const scalar skalar, const scalar skalar_2);

		//Adds two arrays a and b and outputs into result = a *skalar + b *skalar_2 + c *skalar_3
		void Array_Array_Add(const scalar *a, const scalar *b, const scalar *c, scalar *result, const int length, const scalar skalar, const scalar skalar_2, const scalar skalar_3);
		void Array_Array_Add(const std::vector<scalar> &a, const std::vector<scalar> &b, const std::vector<scalar> &c, std::vector<scalar> &result, const scalar skalar, const scalar skalar_2, const scalar skalar_3);

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
		scalar Dot_Product(const scalar *v1, const scalar *v2);
		scalar Dot_Product(const std::vector<scalar> &v1, const std::vector<scalar> &v2);
		scalar Dot_Product(const scalar *v1, const scalar * const* v2, const int b1);
		scalar Dot_Product(const scalar * const* v1, const int a1, const scalar * const* v2, const int b1);
		scalar Dot_Product(const scalar * const* v1, const int a1, const scalar *v2);
		scalar Dot_Product(const scalar * const* const* v1, const int a1, const int a2, const scalar *v2);

		//Calculates the Cross product of v1 and v2 and write it into v3
		void Cross_Product(const scalar *v1, const scalar *v2, scalar *v3);
		void Cross_Product(const scalar *v1, const std::vector<scalar> & v2, scalar *v3);
		void Cross_Product(const std::vector<scalar> &v1, const std::vector<scalar> &v2, std::vector<scalar> &v3);
		void Cross_Product(const scalar *const* v1, const int a1, const scalar *const* v2, const int b1, scalar *v3);
		void Cross_Product(const scalar *const* v1, const int a1, const scalar *const* const* v2, const int b1, const int b2, scalar *v3);

		//Normalizes an arbitrary vector of scalars with its Cartesian Length
		template <typename T>
		void Normalize(T *array, const int length);
		void Normalize(std::vector<scalar> &vec);
		// Normalize each 3-component subvector of a 3N-dimensional vector
		void Normalize_3Nos(std::vector<scalar> &vec);

		//Normalizes an array of array[dim][no_v] of vectors with their respective Kartesian Length
		void Normalize(const int dim, const int no_v, scalar **array);

		//Calculates the angle between two 3D vectors
		// degrees for unit==true; radians for unit==false
		scalar Angle(const scalar *v1, const scalar *v2, bool unit);
		scalar Angle(const std::vector<scalar> &v1, const std::vector<scalar> &v2, bool unit);

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
#include "Vectormath.hxx"
#endif