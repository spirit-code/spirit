#ifndef MANAGED_ALLOCATOR_H
#define MANAGED_ALLOCATOR_H

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>

// static void HandleError( cudaError_t err, const char *file, int line )
// {
// 	// CUDA error handeling from the "CUDA by example" book
// 	if (err != cudaSuccess)
//   {
// 		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
// 		exit( EXIT_FAILURE );
// 	}
// }

// #define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


template<class T>
class managed_allocator
{
  public:
    using value_type = T;
  
    value_type* allocate(size_t n)
    {
      value_type* result = nullptr;

      cudaError_t err = cudaMallocManaged(&result, n*sizeof(T), cudaMemAttachGlobal);
      if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
      }

      return result;
    }
  
    void deallocate(value_type* ptr, size_t)
    {
      cudaError_t err = cudaFree(ptr);
      if (err != cudaSuccess)
      {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), __FILE__, __LINE__ );
        exit( EXIT_FAILURE );
      }
    }
};

#endif
#endif