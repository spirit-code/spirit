#include "Vectormath.h"
#include <vector>

namespace Utility
{
	namespace Vectormath
	{
		//====================================== VECTOR COPY =======================================
		// Copies Vector b of dimension dim into a
		template <typename T1> void Vector_Copy(T1 * a, const T1 *b, const int dim) {
			for (int i = 0; i < dim; ++i) {
				a[i] = b[i];
			}//endfor i
		}
		template<typename T1>
		void Vector_Copy_io(std::vector<T1> & a, const std::vector<T1> & b)
		{
			for (int i = 0; i < (int)b.size(); ++i) {
				a[i] = b[i];
			}//endfor i
		}
		//end Vector_Copy 1d

		 // For arg=0	copies Vector	b[:][pos_x] into a
		 // Else		copies Vector	b[pos_x][:] into a
		template <typename T1> void Vector_Copy(T1 *a, const T1 *const *b, const int dim, const int arg, const int pos_x) {
			if (arg == 0) {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[i][pos_x];
				}//endfor i
			}
			else {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[pos_x][i];
				}//endfor i
			}
		}
		template<typename T1>
		void Vector_Copy_o(std::vector<T1> & a, const std::vector<std::vector<T1>> & b, const int arg, const int pos_x)
		{
			if (arg == 0) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[i] = b[i][pos_x];
				}//endfor i
			}
			else {
				for (int i = 0; i < (int)b[pos_x].size(); ++i) {
					a[i] = b[pos_x][i];
				}//endfor i
			}
		}//end Vector_Copy 2d

		 // For arg=0	copies Vector	b[:][pos_x][pos_y] into a
		 // For arg=1	copies Vector	b[pos_x][:][pos_y] into a
		 // Else		copies Vector	b[pos_x][pos_y][:] into a
		template <typename T1> void Vector_Copy(T1 * a, const T1 *const *const *b, const int dim, const int arg, const int pos_x, const int pos_y) {
			if (arg == 0) {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[i][pos_x][pos_y];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[pos_x][i][pos_y];
				}//endfor i
			}
			else {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[pos_x][pos_y][i];
				}//endfor i
			}
		}
		template <typename T1>
		void Vector_Copy_o(std::vector<T1> & a, const std::vector<std::vector<std::vector<T1>>> &b, const int arg, const int pos_x, const int pos_y) {
			if (arg == 0) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[i] = b[i][pos_x][pos_y];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < (int)b[pos_x].size(); ++i) {
					a[i] = b[pos_x][i][pos_y];
				}//endfor i
			}
			else {
				for (int i = 0; i < (int)b[pos_x][pos_y].size(); ++i) {
					a[i] = b[pos_x][pos_y][i];
				}//endfor i
			}
		}//end Vector_Copy 3d

		 // For arg=0	copies Vector	b[:][pos_x][pos_y][pos_z] into a
		 // For arg=1	copies Vector	b[pos_x][:][pos_y][pos_z] into a
		 // For arg=2	copies Vector	b[pos_x][pos_y][:][pos_z] into a
		 // Else		copies Vector	b[pos_x][pos_y][pos_z][:] into a
		template <typename T1> void Vector_Copy(T1 * a, const T1 *const *const *const *b, const int dim, const int arg, const int pos_x, const int pos_y, const int pos_z) {
			if (arg == 0) {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[i][pos_x][pos_y][pos_z];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[pos_x][i][pos_y][pos_z];
				}//endfor i
			}
			else if (arg == 2) {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[pos_x][pos_y][i][pos_z];
				}//endfor i
			}
			else {
				for (int i = 0; i < dim; ++i) {
					a[i] = b[pos_x][pos_y][pos_z][i];
				}//endfor i
			}
		}
		template <typename T1>
		void Vector_Copy_o(std::vector<T1> & a, const std::vector<std::vector<std::vector<std::vector<T1>>>> &b, const int arg, const int pos_x, const int pos_y, const int pos_z) {
			if (arg == 0) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[i] = b[i][pos_x][pos_y][pos_z];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < (int)b[pos_x].size(); ++i) {
					a[i] = b[pos_x][i][pos_y][pos_z];
				}//endfor i
			}
			else if (arg == 2) {
				for (int i = 0; i < (int)b[pos_x][pos_y].size(); ++i) {
					a[i] = b[pos_x][pos_y][i][pos_z];
				}//endfor i
			}
			else {
				for (int i = 0; i < (int)b[pos_x][pos_y][pos_z].size(); ++i) {
					a[i] = b[pos_x][pos_y][pos_z][i];
				}//endfor i
			}
		}//end Vector Copy 4d

		 // For arg=0	copies Vector	b into a[:][pos_x]
		 // Else		copies Vector	b into a[pos_x][:]
		template <typename T1> void Vector_Copy(T1 **a, const T1 *b, const int dim, const int arg, const int pos_x) {
			if (arg == 0) {
				for (int i = 0; i < dim; ++i) {
					a[i][pos_x] = b[i];
				}//endfor i
			}
			else {
				for (int i = 0; i < dim; ++i) {
					a[pos_x][i] = b[i];
				}//endfor i
			}
		}
		template <typename T1> void Vector_Copy_i(std::vector<std::vector<T1>> & a, const std::vector<T1> &b, const int arg, const int pos_x) {
			if (arg == 0) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[i][pos_x] = b[i];
				}//endfor i
			}
			else {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[pos_x][i] = b[i];
				}//endfor i
			}
		}//end Vector_Copy 2d

		 // For arg=0	copies Vector	b into a[:][pos_x][pos_y]
		 // For arg=1	copies Vector	b into a[pos_x][:][pos_y]
		 // Else		copies Vector	b into a[pos_x][pos_y][:]
		template <typename T1> void Vector_Copy(T1 *** a, const T1 *b, const int dim, const int arg, const int pos_x, const int pos_y) {
			if (arg == 0) {
				for (int i = 0; i < dim; ++i) {
					a[i][pos_x][pos_y] = b[i];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < dim; ++i) {
					a[pos_x][i][pos_y] = b[i];
				}//endfor i
			}
			else {
				for (int i = 0; i < dim; ++i) {
					a[pos_x][pos_y][i] = b[i];
				}//endfor i
			}
		}
		template <typename T1> void Vector_Copy_i(std::vector<std::vector<std::vector<T1>>> &a, const std::vector<T1> &b, const int arg, const int pos_x, const int pos_y) {
			if (arg == 0) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[i][pos_x][pos_y] = b[i];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[pos_x][i][pos_y] = b[i];
				}//endfor i
			}
			else {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[pos_x][pos_y][i] = b[i];
				}//endfor i
			}
		}//end Vector_Copy 3d

		 // For arg=0	copies Vector	b into a[:][pos_x][pos_y][pos_z]
		 // For arg=1	copies Vector	b into a[pos_x][:][pos_y][pos_z]
		 // For arg=2	copies Vector	b into a[pos_x][pos_y][:][pos_z]
		 // Else		copies Vector	b into a[pos_x][pos_y][pos_z][:]
		template <typename T1> void Vector_Copy(T1 **** a, const T1 *b, const int dim, const int arg, const int pos_x, const int pos_y, const int pos_z) {
			if (arg == 0) {
				for (int i = 0; i < dim; ++i) {
					a[i][pos_x][pos_y][pos_z] = b[i];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < dim; ++i) {
					a[pos_x][i][pos_y][pos_z] = b[i];
				}//endfor i
			}
			else if (arg == 2) {
				for (int i = 0; i < dim; ++i) {
					a[pos_x][pos_y][i][pos_z] = b[i];
				}//endfor i
			}
			else {
				for (int i = 0; i < dim; ++i) {
					a[pos_x][pos_y][pos_z][i] = b[i];
				}//endfor i
			}
		}
		template <typename T1> void Vector_Copy_i(std::vector<std::vector<std::vector<std::vector<T1>>>> &a, const std::vector<T1> &b, const int arg, const int pos_x, const int pos_y, const int pos_z) {
			if (arg == 0) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[i][pos_x][pos_y][pos_z] = b[i];
				}//endfor i
			}
			else if (arg == 1) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[pos_x][i][pos_y][pos_z] = b[i];
				}//endfor i
			}
			else if (arg == 2) {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[pos_x][pos_y][i][pos_z] = b[i];
				}//endfor i
			}
			else {
				for (int i = 0; i < (int)b.size(); ++i) {
					a[pos_x][pos_y][pos_z][i] = b[i];
				}//endfor i
			}
		}//end Vector Copy 4d

		// Copies Array b of dimensions dim_w, dim_x into a
		template <typename T1> void Array_Copy(T1 ** a, const T1* const *b, const int dim_w, const int dim_x) {
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					a[i][j] = b[i][j];
				}
			}
		}//end Array_Copy 2d

		//============================= ALLOCATING RECTANGULAR ARRAYS ==============================
		template <typename T1> void Alloc_Array(T1 * &a, const int dim_w) {
			a = new T1[dim_w];
			memset(a, 0, dim_w*sizeof(T1));
		}//end 1d-Alloc_Array

		template <typename T1> void Alloc_Array(T1 ** &a, const int dim_w, const int dim_x) {
			a = new T1*[dim_w];
			for (int i = 0; i < dim_w; ++i) {
				a[i] = new T1[dim_x];
				memset(a[i], 0, dim_x*sizeof(T1));
			}//endfor i
		}//end 2d-Alloc_Array

		template <typename T1> void Alloc_Array(T1 *** &a, const int dim_w, const int dim_x, const int dim_y) {
			a = new T1**[dim_w];
			for (int i = 0; i < dim_w; ++i) {
				a[i] = new T1*[dim_x];
				for (int j = 0; j < dim_x; ++j) {
					a[i][j] = new T1[dim_y];
					memset(a[i][j], 0, dim_y*sizeof(T1));
				}//endfor j
			}//endfor i
		}//end 3d-Alloc_Array

		template <typename T1> void Alloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int dim_y, const int dim_z) {
			a = new T1***[dim_w];
			for (int i = 0; i < dim_w; ++i) {
				a[i] = new T1**[dim_x];
				for (int j = 0; j < dim_x; ++j) {
					a[i][j] = new T1*[dim_y];
					for (int k = 0; k < dim_y; ++k) {
						a[i][j][k] = new T1[dim_z];
						memset(a[i][j][k], 0, dim_z*sizeof(T1));
					}//endfor k
				}//endfor j
			}//endfor i
		}//end 4d-Alloc_Array
		//=========================== END ALLOCATING RECTANGULAR ARRAYS ============================
		//================================ COPY RECTANGULAR ARRAYS =================================
		template <typename T1> void Copy_Array(T1 * &a, T1 *b, const int dim_w) {
			Alloc_Array(a, dim_w);
			for (int i = 0; i < dim_w; ++i) {
				a[i] = b[i];
			}// endfor i
		}// end 1-d-Copy_Array
		template <typename T1> void Copy_Array(T1 ** &a, T1 **b, const int dim_w, const int dim_x) {
			Alloc_Array(a, dim_w, dim_x);
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					a[i][j] = b[i][j];
				}// endfor j
			}// endfor i
		}// end 2-d-Copy_Array
		template <typename T1> void Copy_Array(T1 *** &a, T1 ***b, const int dim_w, const int dim_x, const int dim_y) {
			Alloc_Array(a, dim_w, dim_x, dim_y);
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					for (int k = 0; k < dim_y; ++k) {
						a[i][j][k] = b[i][j][k];
					}// endfor k
				}// endfor j
			}// endfor i
		}// end 3-d-Copy_Array
		template <typename T1> void Copy_Array(T1 **** &a, T1 ****b, const int dim_w, const int dim_x, const int dim_y, const int dim_z) {
			Alloc_Array(a, dim_w, dim_x, dim_y, dim_z);
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					for (int k = 0; k < dim_y; ++k) {
						for (int l = 0; l < dim_z; ++l) {
							a[i][j][k][l] = b[i][j][k][l];
						}// endfor l
					}// endfor k
				}// endfor j
			}// endfor i
		}// end 3-d-Copy_Array

		//============================== END COPY RECTANGULAR ARRAYS ===============================
		//============================ DEALLOCATING RECTANGULAR ARRAYS =============================
		template <typename T1>	void Dealloc_Array(T1 * &a) {
			delete[] a;
		}//end 1d-Dealloc

		template <typename T1> void Dealloc_Array(T1 ** &a, const int dim_w) {
			for (int i = 0; i < dim_w; ++i) {
				delete[] a[i];
			}//endfor i
			delete[] a;
		}//end 2d-Dealloc

		template <typename T1> void Dealloc_Array(T1 *** &a, const int dim_w, const int dim_x) {
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					delete[] a[i][j];
				}//endfor j
				delete[] a[i];
			}//endfor i
			delete[] a;
		}//end 3d-Dealloc

		template <typename T1> void Dealloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int dim_y) {
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					for (int k = 0; k < dim_y; ++k) {
						delete[] a[i][j][k];
					}//endfor k
					delete[] a[i][j];
				}//endfor j
				delete[] a[i];
			}//endfor i
			delete[] a;
		}//end 4d-Dealloc

		//========================== END DEALLOCATING RECTANGULAR ARRAYS ===========================
		//============================== ALLOCATING DYNAMIC ARRAYS =================================
		// 3D array dyn with dim_y[dim_x]
		template <typename T1> void Alloc_Array(T1 *** &a, const int dim_w, const int dim_x, const int *const &dim_y) {
			a = new T1**[dim_w];
			for (int i = 0; i < dim_w; ++i) {
				a[i] = new T1*[dim_x];
				for (int j = 0; j < dim_x; ++j) {
					a[i][j] = new T1[dim_y[j]];
					memset(a[i][j], 0, dim_y[j] * sizeof(T1));
				}//endfor j
			}//endfor i
		}//end 3d-Alloc_Array_Dyn
		template <typename T1> void Copy_Array(T1 *** &a, T1 *** b, const int dim_w, const int dim_x, const int *const &dim_y) {
			Alloc_Array(a, dim_w, dim_x, dim_y);
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					for (int k = 0; k < dim_y[j]; ++k) {
						a[i][j][k] = b[i][j][k];
					}// endfor k
				}// endfor j
			}// endfor i
		}// end 3d-Copy_array_Dyn

		// 4D array dyn with dim_y[dim_x]
		template <typename T1> void Alloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int dim_y, const int *const &dim_z) {
			a = new T1***[dim_w];
			for (int i = 0; i < dim_w; ++i) {
				a[i] = new T1**[dim_x];
				for (int j = 0; j < dim_x; ++j) {
					a[i][j] = new T1*[dim_y];
					for (int k = 0; k < dim_y; ++k) {
						a[i][j][k] = new T1[dim_z[k]];
						memset(a[i][j][k], 0, dim_z[k] * sizeof(T1));
					}//endfor k
				}//endfor j
			}//endfor i
		}//end 4d_Alloc_Array_Dyn
		template <typename T1> void Copy_Array(T1 **** &a, T1 ****b, const int dim_w, const int dim_x, const int dim_y, const int *const &dim_z) {
			Alloc_Array(a, dim_w, dim_x, dim_y, dim_z);
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					for (int k = 0; k < dim_y; ++k) {
						for (int l = 0; l < dim_z[k]; ++l) {
							a[i][j][k][l] = b[i][j][k][l];
						}// endfor l
					}// endfor k
				}// endfor j
			}// endfor i
		}


		//============================ END ALLOCATING DYNAMIC ARRAYS ================================


		//============================= DEALLOCATING DYNAMIC ARRAYS =================================
		// 4D array dyn with dim_y[dim_x]
		template <typename T1> void Dealloc_Array(T1 **** &a, const int dim_w, const int dim_x, const int *const &dim_y) {
			for (int i = 0; i < dim_w; ++i) {
				for (int j = 0; j < dim_x; ++j) {
					for (int k = 0; k < dim_y[j]; ++k) {
						delete[] a[i][j][k];
					}//endfor k
					delete[] a[i][j];
				}//endfor j
				delete[] a[i];
			}//endfor i
			delete[] a;
		}//end 4d-Dealloc-Dyn
		//=========================== END DEALLOCATING DYNAMIC ARRAYS ==============================

	}//end namespace Vectormath
}// end namespace Utility