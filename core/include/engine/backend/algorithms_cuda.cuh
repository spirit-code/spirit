#pragma once

#ifdef SPIRIT_USE_CUDA
#include <engine/backend/types.hpp>

#include <engine/backend/Transform_Iterator.hpp>
#include <engine/backend/Zip_Iterator.hpp>

#include <cub/cub.cuh>

#include <algorithm>
#include <numeric>

/*
 * NOTE: The `device_iterator_cast` and the overloads for `is_device_iterator` and `is_host_iterator` are more of a
 *       hack rather than a proper solution.
 *       I tried building the backend on top of `thrust::universal_vector`, but that fails because these cannot be
 *       stored in 'CUDA managed memory' to my best attempts.
 *       The simplest approach for maintainability would probably be implementing a custom vector class that is
 *       `__host__ __device__` enabled; can be stored in universal memory; and most importantly produces device enabled
 *       iterators. This would also make the workaround of having device enabled `spans` over the currently used
 *       `std::vectors` obsolete.
 *       The best solution for performance considerations would certainly be to flatten the index data structure from
 *       the Hamiltonian (the only place where storing a device enabled vector inside a device enabled vector is
 *       strictly needed at the moment). The main drawback to that approach is that this either requires a device
 *       enabled type erased container type (like std::any) to store the different types of indices in or
 *       one alternatively could flatten the whole structure into a `vector<int>` (or similar) and access the
 *       unerlying data using a second vector of offsets (similar to a CSR matrix encoding) over which the transform
 *       operation would be performed.
 *       The first implementation of the type erased container could prove challenging, because as far as I'm aware no
 *       such implementation currently exists in any of the publically available CUDA libraries; the second
 *       implementation has the drawback of limiting the possible design space for index types, but most of them would
 *       probably be representable (if not in the most space efficient way) by integer sequences as suggested before.
 */

namespace Engine
{

namespace Backend
{

namespace cuda
{

namespace detail
{

// all device capable iterators used in the code need to be added here
template<typename Iter>
struct is_device_iterator : std::false_type
{
};

template<typename T>
struct is_device_iterator<T *> : std::true_type
{
};

template<typename Incrementable>
struct is_device_iterator<Backend::cuda::counting_iterator<Incrementable>> : std::true_type
{
};

template<typename InputIt, typename UnaryOp, typename Value>
struct is_device_iterator<Backend::cuda::transform_iterator<Value, UnaryOp, InputIt>> : std::true_type
{
};

template<typename IteratorTuple>
struct is_device_iterator<Backend::cuda::zip_iterator<IteratorTuple>> : std::true_type
{
};

} // namespace detail

// cast the iterator to a device iterator (casting normal iterators to raw pointers using `raw_pointer_cast`)
template<typename Iter>
auto device_iterator_cast( Iter it )
{
    if constexpr( detail::is_device_iterator<std::decay_t<Iter>>::value )
        return it;
    else
    {
        return raw_pointer_cast( it );
    }
}

// TODO: replace most of these overloads by cub implementations (like cub::DeviceFor) once these become widely available

// ===== copy ==========================================================================================================

namespace detail
{

namespace seq
{

template<class InputIt, class OutputIt>
__device__ __forceinline__ OutputIt copy( InputIt first, InputIt last, OutputIt d_first )
{
    while( first != last )
        *( d_first++ ) = *( first++ );

    return d_first;
}

template<class InputIt, class OutputIt>
__device__ __forceinline__ OutputIt copy_n( InputIt first, const int n, OutputIt d_first )
{
    for( int i = 0; i < n; ++i )
        *( d_first++ ) = *( first++ );

    return d_first;
}

} // namespace seq

namespace kernel
{

template<class InputIt, class OutputIt>
__global__ void copy_n( InputIt first, const int n, OutputIt d_first )
{
    static_assert( is_device_iterator<InputIt>::value );
    static_assert( is_device_iterator<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n )
        d_first[idx] = first[idx];
}

} // namespace kernel

namespace par
{

template<class InputIt, class OutputIt>
__host__ OutputIt copy_n( InputIt first, const int n, OutputIt d_first )
{
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::copy_n<
                std::decay_t<decltype( device_iterator_cast( first ) )>,
                std::decay_t<decltype( device_iterator_cast( d_first ) )>> ) );
        return blockSize;
    }();
    Backend::cuda::detail::kernel::copy_n<<<( n + blockSize - 1 ) / blockSize, blockSize>>>(
        device_iterator_cast( first ), n, device_iterator_cast( d_first ) );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, n );
}

template<class InputIt, class OutputIt>
__host__ OutputIt copy( InputIt first, InputIt last, OutputIt d_first )
{
    return Backend::cuda::detail::par::copy_n( first, std::distance( first, last ), d_first );
}

} // namespace par

} // namespace detail

template<class InputIt, class OutputIt, class UnaryOp>
__host__ __device__ auto copy( InputIt first, InputIt last, OutputIt && d_first ) -> OutputIt
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::copy( first, last, std::forward<OutputIt>( d_first ) );
}

template<class InputIt, class OutputIt>
__host__ auto copy( const ::execution::cuda::par_t &, InputIt first, InputIt last, OutputIt && d_first ) -> OutputIt
{
    return Backend::cuda::detail::par::copy( first, last, std::forward<OutputIt>( d_first ) );
}

template<class InputIt, class Size, class OutputIt, class UnaryOp>
__host__ __device__ auto copy_n( InputIt && first, Size n, OutputIt && d_first ) -> OutputIt
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::copy_n( std::forward<InputIt>( first ), n, std::forward<OutputIt>( d_first ) );
}

template<class InputIt, class Size, class OutputIt>
__host__ auto copy_n( const ::execution::cuda::par_t &, InputIt && first, Size n, OutputIt && d_first ) -> OutputIt
{
    return Backend::cuda::detail::par::copy_n( std::forward<InputIt>( first ), n, std::forward<OutputIt>( d_first ) );
}

// ===== fill ==========================================================================================================

namespace detail
{

namespace seq
{

template<class InputIt, class T>
__device__ __forceinline__ void fill( InputIt first, InputIt last, const T & value )
{
    while( first != last )
        *( first++ ) = value;
}

template<class InputIt, class T>
__device__ __forceinline__ InputIt fill_n( InputIt first, const int n, const T & value )
{
    for( int i = 0; i < n; ++i )
        *( first++ ) = value;

    return first;
}

} // namespace seq

namespace kernel
{

template<class InputIt, class T>
__global__ void fill_n( InputIt first, const int n, const T value )
{
    static_assert( is_device_iterator<InputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n )
    {
        first[idx] = value;
    }
}

} // namespace kernel

namespace par
{

template<class InputIt, class T>
__host__ InputIt fill_n( InputIt first, const int n, const T & value )
{
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::fill_n<std::decay_t<decltype( device_iterator_cast( first ) )>, T> ) );
        return blockSize;
    }();
    Backend::cuda::detail::kernel::fill_n<<<( n + blockSize - 1 ) / blockSize, blockSize>>>(
        device_iterator_cast( first ), n, value );
    CU_CHECK_AND_SYNC();
    return std::next( first, n );
}

template<class InputIt, class T>
__host__ InputIt fill( InputIt first, InputIt last, const T & value )
{
    return Backend::cuda::detail::par::fill_n( first, std::distance( first, last ), value );
}

} // namespace par

} // namespace detail

template<class InputIt, class T>
__host__ __device__ auto fill( InputIt first, InputIt last, const T & value ) -> void
{
#ifdef __CUDA_ARCH__
    Backend::cuda::detail::seq
#else
    std
#endif
    ::fill( first, last, value );
}

template<class InputIt, class T>
__host__ auto fill( const ::execution::cuda::par_t &, InputIt first, InputIt last, const T & value ) -> void
{
    Backend::cuda::detail::par::fill( first, last, value );
}

template<class InputIt, class Size, class T>
__host__ __device__ auto fill_n( InputIt first, Size n, const T & value ) -> InputIt
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::fill_n( first, n, value );
}

template<class InputIt, class Size, class T>
__host__ auto fill_n( const ::execution::cuda::par_t &, InputIt first, Size n, const T & value ) -> InputIt
{
    return Backend::cuda::detail::par::fill_n( first, n, value );
}

// ===== for_each ======================================================================================================

namespace detail
{

namespace seq
{

template<class InputIt, class UnaryOp>
__device__ __forceinline__ UnaryOp for_each( InputIt first, InputIt last, UnaryOp unary_op )
{
    static_assert( is_device_iterator<InputIt>::value );

    while( first != last )
        unary_op( *( first++ ) );

    return unary_op;
}

template<class InputIt, class Size, class UnaryOp>
__device__ __forceinline__ InputIt for_each_n( InputIt first, Size n, UnaryOp unary_op )
{
    static_assert( is_device_iterator<InputIt>::value );

    for( Size i = 0; i < n; ++i )
        unary_op( *( first++ ) );

    return first;
}

} // namespace seq

namespace kernel
{

template<class InputIt, class UnaryOp>
__global__ void for_each_n( InputIt first, const int n, UnaryOp unary_op )
{
    static_assert( is_device_iterator<InputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n )
        unary_op( first[idx] );
}

} // namespace kernel

namespace par
{

template<class InputIt, class UnaryOp>
__host__ InputIt for_each_n( InputIt first, const int n, UnaryOp unary_op )
{
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::for_each_n<
                std::decay_t<decltype( device_iterator_cast( first ) )>, UnaryOp> ) );
        return blockSize;
    }();
    Backend::cuda::detail::kernel::for_each_n<<<( n + blockSize - 1 ) / blockSize, blockSize>>>(
        device_iterator_cast( first ), n, unary_op );
    CU_CHECK_AND_SYNC();
    return std::next( first, n );
}

template<class InputIt, class UnaryOp>
__host__ void for_each( InputIt first, InputIt last, UnaryOp && unary_op )
{
    Backend::cuda::detail::par::for_each_n( first, std::distance( first, last ), std::forward<UnaryOp>( unary_op ) );
}

} // namespace par

} // namespace detail

template<class InputIt, class UnaryOp>
__host__ __device__ auto for_each( InputIt first, InputIt last, UnaryOp unary_op ) -> UnaryOp
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::for_each( first, last, unary_op );
}

template<class InputIt, class UnaryOp>
__host__ auto for_each( const ::execution::cuda::par_t &, InputIt first, InputIt last, UnaryOp unary_op ) -> void
{
    Backend::cuda::detail::par ::for_each( first, last, unary_op );
}

template<class InputIt, class Size, class UnaryOp>
__host__ __device__ auto for_each_n( InputIt && first, Size n, UnaryOp && unary_op ) -> InputIt
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::for_each_n( std::forward<InputIt>( first ), n, std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt, class Size, class UnaryOp>
__host__ auto for_each_n( const ::execution::cuda::par_t &, InputIt && first, Size n, UnaryOp && unary_op ) -> InputIt
{
    return Backend::cuda::detail::par ::for_each_n(
        std::forward<InputIt>( first ), n, std::forward<UnaryOp>( unary_op ) );
}

// ===== reduce ========================================================================================================

namespace detail
{

namespace seq
{

template<class InputIt, class OutputIt, class T, class BinaryOp>
__device__ __forceinline__ auto reduce( InputIt first, InputIt last, T init, BinaryOp binary_op ) -> T
{
    static_assert( is_device_iterator<InputIt>::value );

    while( first != last )
        init = binary_op( init, *( first++ ) );

    return init;
};

template<class InputIt>
__device__ __forceinline__ auto reduce( InputIt first, InputIt last ) ->
    typename std::iterator_traits<InputIt>::value_type
{
    static_assert( is_device_iterator<InputIt>::value );

    auto init = std::iterator_traits<InputIt>::value_type();
    while( first != last )
        init += *( first++ );

    return init;
};

} // namespace seq

} // namespace detail

template<class InputIt, class T, class BinaryOp>
__host__ auto reduce( const ::execution::cuda::par_t &, InputIt first, InputIt last, T init, BinaryOp binary_op ) -> T
{
    const int N = std::distance( first, last );

    field<T> ret( 1, init );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, device_iterator_cast( first ), ret.data(), N, binary_op, init );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Reduce(
        d_temp_storage, temp_storage_bytes, device_iterator_cast( first ), ret.data(), N, binary_op, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<class InputIt>
__host__ auto reduce( const ::execution::cuda::par_t &, InputIt first, InputIt last ) ->
    typename std::iterator_traits<InputIt>::value_type
{
    const int N = std::distance( first, last );

    field<typename std::iterator_traits<InputIt>::value_type> ret( 1 );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, device_iterator_cast( first ), ret.data(), N );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, device_iterator_cast( first ), ret.data(), N );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<class InputIt, class T, class BinaryOp>
__host__ auto reduce( InputIt first, InputIt last, T init, BinaryOp binary_op ) -> T
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::reduce( first, last, init, binary_op );
}

template<class InputIt>
__host__ auto reduce( InputIt first, InputIt last ) -> typename std::iterator_traits<InputIt>::value_type
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::reduce( first, last );
}

// ===== transform =====================================================================================================
namespace detail
{

namespace seq
{

template<class InputIt, class OutputIt, class UnaryOp>
__device__ __forceinline__ OutputIt transform( InputIt first, InputIt last, OutputIt d_first, UnaryOp unary_op )
{
    while( first != last )
        *( d_first++ ) = unary_op( *( first++ ) );

    return d_first;
};

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__device__ __forceinline__ OutputIt
transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    while( first1 != last1 )
        *( d_first++ ) = binary_op( *( first1++ ), *( first2++ ) );

    return d_first;
}

} // namespace seq

namespace kernel
{

template<class InputIt, class OutputIt, class UnaryOp>
__global__ void transform_n( InputIt first, int n, OutputIt d_first, UnaryOp unary_op )
{
    static_assert( is_device_iterator<InputIt>::value );
    static_assert( is_device_iterator<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n )
        d_first[idx] = unary_op( first[idx] );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__global__ void transform_n( InputIt1 first1, int n, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    static_assert( is_device_iterator<InputIt1>::value );
    static_assert( is_device_iterator<InputIt2>::value );
    static_assert( is_device_iterator<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n )
        d_first[idx] = binary_op( first1[idx], first2[idx] );
}

} // namespace kernel

namespace par
{

template<class InputIt, class OutputIt, class UnaryOp>
__host__ OutputIt transform( InputIt first, InputIt last, OutputIt d_first, UnaryOp unary_op )
{
    const int n                = std::distance( first, last );
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::transform_n<
                std::decay_t<decltype( device_iterator_cast( first ) )>,
                std::decay_t<decltype( device_iterator_cast( d_first ) )>, UnaryOp> ) );
        return blockSize;
    }();
    Backend::cuda::detail::kernel::transform_n<<<( n + blockSize - 1 ) / blockSize, blockSize>>>(
        device_iterator_cast( first ), n, device_iterator_cast( d_first ), unary_op );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, n );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    const int n                = std::distance( first1, last1 );
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::transform_n<
                std::decay_t<decltype( device_iterator_cast( first1 ) )>,
                std::decay_t<decltype( device_iterator_cast( first2 ) )>,
                std::decay_t<decltype( device_iterator_cast( d_first ) )>, BinaryOp> ) );
        return blockSize;
    }();

    Backend::cuda::detail::kernel::transform_n<<<( n + blockSize - 1 ) / blockSize, blockSize>>>(
        device_iterator_cast( first1 ), n, device_iterator_cast( first2 ), device_iterator_cast( d_first ), binary_op );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, n );
}

} // namespace par

} // namespace detail

template<class InputIt, class OutputIt, class UnaryOp>
__host__ __device__ auto transform( InputIt first, InputIt last, OutputIt && d_first, UnaryOp && unary_op ) -> OutputIt
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::transform( first, last, std::forward<OutputIt>( d_first ), std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt, class OutputIt, class UnaryOp>
__host__ __device__ auto
transform( const ::execution::cuda::par_t &, InputIt first, InputIt last, OutputIt && d_first, UnaryOp && unary_op )
    -> OutputIt
{
    return Backend::cuda::detail::par ::transform(
        first, last, std::forward<OutputIt>( d_first ), std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ __device__ auto
transform( InputIt1 first1, InputIt1 last1, InputIt2 && first2, OutputIt && d_first, BinaryOp && binary_op ) -> OutputIt
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::transform(
            first1, last1, std::forward<InputIt2>( first2 ), std::forward<OutputIt>( d_first ),
            std::forward<BinaryOp>( binary_op ) );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ auto transform(
    const ::execution::cuda::par_t &, InputIt1 first1, InputIt1 last1, InputIt2 && first2, OutputIt && d_first,
    BinaryOp && binary_op ) -> OutputIt
{
    return Backend::cuda::detail::par ::transform(
        first1, last1, std::forward<InputIt2>( first2 ), std::forward<OutputIt>( d_first ),
        std::forward<BinaryOp>( binary_op ) );
}

// ===== transform_reduce ==============================================================================================
namespace detail
{

namespace seq
{

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__device__ __forceinline__ T
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    static_assert( is_device_iterator<InputIt>::value );

    while( first != last )
        init = reduce( init, transform( *( first++ ) ) );

    return init;
}

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__device__ __forceinline__ T transform_reduce(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryReductionOp reduce, BinaryTransformOp transform )
{
    static_assert( is_device_iterator<InputIt1>::value );
    static_assert( is_device_iterator<InputIt2>::value );

    while( first1 != last1 )
        init = reduce( init, transform( *( first1++ ), *( first2++ ) ) );

    return init;
}

} // namespace seq

namespace par
{

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ T transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    const int n = std::distance( first, last );

    field<T> ret( 1, init );

    auto t_first = Backend::cuda::make_transform_iterator( device_iterator_cast( first ), transform );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), n, reduce, init );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), n, reduce, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

} // namespace par

} // namespace detail

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ __device__ auto
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp && reduce, UnaryTransformOp && transform ) -> T
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::transform_reduce(
            first, last, init, std::forward<BinaryReductionOp>( reduce ), std::forward<UnaryTransformOp>( transform ) );
};

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ auto transform_reduce(
    const ::execution::cuda::par_t &, InputIt first, InputIt last, T init, BinaryReductionOp && reduce,
    UnaryTransformOp && transform ) -> T
{
    return Backend::cuda::detail::par::transform_reduce(
        first, last, init, std::forward<BinaryReductionOp>( reduce ), std::forward<UnaryTransformOp>( transform ) );
};

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__host__ __device__ auto transform_reduce(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, T && init, BinaryReductionOp && reduce,
    BinaryTransformOp && transform ) -> T
{
    return
#ifdef __CUDA_ARCH__
        Backend::cuda::detail::seq
#else
        std
#endif
        ::transform_reduce(
            Backend::cuda::make_zip_iterator( first1, first2 ),
            Backend::cuda::make_zip_iterator( last1, first2 + std::distance( first1, last1 ) ), std::forward<T>( init ),
            std::forward<BinaryReductionOp>( reduce ),
            Backend::cuda::make_zip_function( std::forward<BinaryTransformOp>( transform ) ) );
}

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__host__ auto transform_reduce(
    const ::execution::cuda::par_t &, InputIt1 first1, InputIt1 last1, InputIt2 first2, T && init,
    BinaryReductionOp && reduce, BinaryTransformOp && transform ) -> T
{
    return Backend::cuda::detail::par::transform_reduce(
        Backend::cuda::make_zip_iterator( device_iterator_cast( first1 ), device_iterator_cast( first2 ) ),
        Backend::cuda::make_zip_iterator(
            device_iterator_cast( last1 ), device_iterator_cast( first2 ) + std::distance( first1, last1 ) ),
        std::forward<T>( init ), std::forward<BinaryReductionOp>( reduce ),
        Backend::cuda::make_zip_function( std::forward<BinaryTransformOp>( transform ) ) );
}

} // namespace cuda

} // namespace Backend

} // namespace Engine

#endif
