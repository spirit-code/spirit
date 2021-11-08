#ifndef SPIRIT_CORE_ENGINE_MANAGED_ALLOCATOR_HPP
#define SPIRIT_CORE_ENGINE_MANAGED_ALLOCATOR_HPP

#ifdef SPIRIT_USE_CUDA

#include <utility/Exception.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

static void CudaHandleError( cudaError_t err, const char * file, int line, const std::string & function )
{
    if( err != cudaSuccess )
    {
        throw Utility::S_Exception(
            Utility::Exception_Classifier::CUDA_Error, Utility::Log_Level::Severe,
            std::string( cudaGetErrorString( err ) ), file, line, function );
    }
}

#define CU_HANDLE_ERROR( err ) ( CudaHandleError( err, __FILE__, __LINE__, __func__ ) )

#define CU_CHECK_ERROR() ( CudaHandleError( cudaGetLastError(), __FILE__, __LINE__, __func__ ) )

#define CU_CHECK_AND_SYNC()                                                                                            \
    CU_CHECK_ERROR();                                                                                                  \
    CU_HANDLE_ERROR( cudaDeviceSynchronize() )

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

    value_type * allocate( size_t n )
    {
        value_type * result = nullptr;

        CU_HANDLE_ERROR( cudaMallocManaged( &result, n * sizeof( value_type ) ) );

        return result;
    }

    void deallocate( value_type * ptr, size_t )
    {
        CU_HANDLE_ERROR( cudaFree( ptr ) );
    }

    managed_allocator() throw() : std::allocator<T>() {} // fprintf(stderr, "Hello managed allocator!\n"); }
    managed_allocator( const managed_allocator & a ) throw() : std::allocator<T>( a ) {}
    template<class U>
    managed_allocator( const managed_allocator<U> & a ) throw() : std::allocator<T>( a )
    {
    }
    ~managed_allocator() throw() {}
};

#endif
#endif