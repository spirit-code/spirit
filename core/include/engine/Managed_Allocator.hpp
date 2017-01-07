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
class managed_allocator : public std::allocator<T>
{
public:
    using value_type = T;


    template<typename _Tp1>
    struct rebind
    {
        typedef managed_allocator<_Tp1> other;
    };


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


    managed_allocator() throw(): std::allocator<T>() { } //fprintf(stderr, "Hello managed allocator!\n"); }
    managed_allocator(const managed_allocator &a) throw(): std::allocator<T>(a) { }
    template <class U>                    
    managed_allocator(const managed_allocator<U> &a) throw(): std::allocator<T>(a) { }
    ~managed_allocator() throw() { }
};

#endif
#endif