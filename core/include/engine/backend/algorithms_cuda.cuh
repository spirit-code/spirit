#pragma once

#ifdef SPIRIT_USE_CUDA
#include <engine/backend/types.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <cub/cub.cuh>

template<typename Iter>
struct is_host_iterator
        : std::conjunction<
              std::negation<std::is_pointer<Iter>>,
              std::disjunction<
                  std::is_same<Iter, typename field<typename std::iterator_traits<Iter>::value_type>::iterator>,
                  std::is_same<Iter, typename field<typename std::iterator_traits<Iter>::value_type>::const_iterator>,
                  std::is_same<Iter, typename std::vector<typename std::iterator_traits<Iter>::value_type>::iterator>,
                  std::is_same<
                      Iter, typename std::vector<typename std::iterator_traits<Iter>::value_type>::const_iterator>>>
{
};

namespace Engine
{

namespace Backend
{

namespace cuda
{

using thrust::copy;
using thrust::copy_n;
using thrust::fill;
using thrust::fill_n;
using thrust::for_each;
using thrust::for_each_n;

namespace detail
{

namespace device
{

// no parallelization, because as a __device__ function these are called per thread
template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__device__ __forceinline__ T
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    while( first != last )
        init = reduce( init, transform( *( first++ ) ) );

    return init;
};

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__device__ __forceinline__ T transform_reduce(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryReductionOp reduce, BinaryTransformOp transform )
{
    while( first1 != last1 )
        init = reduce( init, transform( *( first1++ ), *( first2++ ) ) );

    return init;
};

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
};

} // namespace device

namespace kernel
{

template<class InputIt, class OutputIt, class UnaryOp>
__global__ void transform_n( InputIt first, int N, OutputIt d_first, UnaryOp unary_op )
{
    static_assert( std::is_pointer<InputIt>::value );
    static_assert( std::is_pointer<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        d_first[idx] = unary_op( first[idx] );
    }
};

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__global__ void transform_n( InputIt1 first1, int N, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    static_assert( std::is_pointer<InputIt1>::value );
    static_assert( std::is_pointer<InputIt2>::value );
    static_assert( std::is_pointer<OutputIt>::value );

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
    {
        d_first[idx] = binary_op( first1[idx], first2[idx] );
    }
};

} // namespace kernel

namespace host
{

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ T transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform )
{
    const int N = std::distance( first, last );

    field<T> ret( 1, init );

    auto t_first = cub::TransformInputIterator<T, UnaryTransformOp, InputIt>( first, transform );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<class InputIt, class OutputIt, class UnaryOp>
__host__ OutputIt transform( InputIt first, InputIt last, OutputIt d_first, UnaryOp unary_op )
{
    const int N                = std::distance( first, last );
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::transform_n<
                std::decay_t<decltype( raw_pointer_cast( first ) )>,
                std::decay_t<decltype( raw_pointer_cast( d_first ) )>, UnaryOp> ) );
        return blockSize;
    }();
    Backend::cuda::detail::kernel::transform_n<<<( N + blockSize - 1 ) / blockSize, blockSize>>>(
        raw_pointer_cast( first ), N, raw_pointer_cast( d_first ), unary_op );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, N );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ OutputIt transform( InputIt1 first1, InputIt1 last1, InputIt2 first2, OutputIt d_first, BinaryOp binary_op )
{
    const int N                = std::distance( first1, last1 );
    static const int blockSize = []()
    {
        int blockSize   = 0;
        int minGridSize = 0;
        CU_HANDLE_ERROR( cudaOccupancyMaxPotentialBlockSize(
            &blockSize, &minGridSize,
            &Backend::cuda::detail::kernel::transform_n<
                std::decay_t<decltype( raw_pointer_cast( first1 ) )>,
                std::decay_t<decltype( raw_pointer_cast( first2 ) )>,
                std::decay_t<decltype( raw_pointer_cast( d_first ) )>, BinaryOp> ) );
        return blockSize;
    }();

    Backend::cuda::detail::kernel::transform_n<<<( N + blockSize - 1 ) / blockSize, blockSize>>>(
        raw_pointer_cast( first1 ), N, raw_pointer_cast( first2 ), raw_pointer_cast( d_first ), binary_op );
    CU_CHECK_AND_SYNC();
    return std::next( d_first, N );
}

} // namespace host

} // namespace detail

template<class InputIt>
__host__ __device__ auto reduce( InputIt first, InputIt last ) -> typename std::iterator_traits<InputIt>::value_type
{
    const int N = std::distance( first, last );

    field<typename std::iterator_traits<InputIt>::value_type> ret( 1 );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, raw_pointer_cast( first ), ret.data(), N );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, raw_pointer_cast( first ), ret.data(), N );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

// requires that `reduce` is a `__host__ __device__` functor
template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ auto
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp reduce, UnaryTransformOp transform ) ->
    typename std::enable_if<is_host_iterator<typename std::decay<InputIt>::type>::value, T>::type
{
    const int N = std::distance( first, last );

    field<T> ret( 1, init );

    using pointer_t = typename std::iterator_traits<InputIt>::pointer;

    auto t_first = cub::TransformInputIterator<T, UnaryTransformOp, pointer_t>( raw_pointer_cast( first ), transform );

    // Determine temporary storage size and allocate
    void * d_temp_storage     = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    cudaMalloc( &d_temp_storage, temp_storage_bytes );
    // Reduction
    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes, t_first, ret.data(), N, reduce, init );
    CU_CHECK_AND_SYNC();
    cudaFree( d_temp_storage );
    return ret[0];
}

template<class InputIt, class T, class BinaryReductionOp, class UnaryTransformOp>
__host__ __device__ auto
transform_reduce( InputIt first, InputIt last, T init, BinaryReductionOp && reduce, UnaryTransformOp && transform ) ->
    typename std::enable_if<!is_host_iterator<typename std::decay<InputIt>::type>::value, T>::type
{
    return Backend::cuda::detail::
#ifdef __CUDA_ARCH__
        device
#else
        host
#endif
        ::transform_reduce(
            first, last, init, std::forward<BinaryReductionOp>( reduce ), std::forward<UnaryTransformOp>( transform ) );
};

template<class InputIt1, class InputIt2, class T, class BinaryReductionOp, class BinaryTransformOp>
__host__ __device__ T transform_reduce(
    InputIt1 first1, InputIt1 last1, InputIt2 first2, T && init, BinaryReductionOp && reduce,
    BinaryTransformOp && transform )
{
    return Backend::cuda::transform_reduce(
        thrust::make_zip_iterator( first1, first2 ),
        thrust::make_zip_iterator( last1, first2 + std::distance( first1, last1 ) ), std::forward<T>( init ),
        std::forward<BinaryReductionOp>( reduce ),
        thrust::make_zip_function( std::forward<BinaryTransformOp>( transform ) ) );
};

template<class InputIt, class OutputIt, class UnaryOp>
__host__ auto transform( InputIt first, InputIt last, OutputIt && d_first, UnaryOp && unary_op ) ->
    typename std::enable_if<is_host_iterator<typename std::decay<InputIt>::type>::value, OutputIt>::type
{
    return Backend::cuda::detail::host::transform(
        first, last, std::forward<OutputIt>( d_first ), std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt, class OutputIt, class UnaryOp>
__host__ __device__ auto transform( InputIt first, InputIt last, OutputIt && d_first, UnaryOp && unary_op ) ->
    typename std::enable_if<!is_host_iterator<typename std::decay<InputIt>::type>::value, OutputIt>::type
{
    return Backend::cuda::detail::
#ifdef __CUDA_ARCH__
        device
#else
        host
#endif
        ::transform( first, last, std::forward<OutputIt>( d_first ), std::forward<UnaryOp>( unary_op ) );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ auto
transform( InputIt1 first1, InputIt1 last1, InputIt2 && first2, OutputIt && d_first, BinaryOp && binary_op ) ->
    typename std::enable_if<
        std::conjunction<
            is_host_iterator<typename std::decay<InputIt1>::type>,
            is_host_iterator<typename std::decay<InputIt2>::type>>::value,
        OutputIt>::type
{
    return Backend::cuda::detail::host::transform(
        first1, last1, std::forward<InputIt2>( first2 ), std::forward<OutputIt>( d_first ),
        std::forward<BinaryOp>( binary_op ) );
}

template<class InputIt1, class InputIt2, class OutputIt, class BinaryOp>
__host__ __device__ auto
transform( InputIt1 first1, InputIt1 last1, InputIt2 && first2, OutputIt && d_first, BinaryOp && binary_op ) ->
    typename std::enable_if<
        !std::conjunction<
            is_host_iterator<typename std::decay<InputIt1>::type>,
            is_host_iterator<typename std::decay<InputIt2>::type>>::value,
        OutputIt>::type
{
    return Backend::cuda::detail::
#ifdef __CUDA_ARCH__
        device
#else
        host
#endif
        ::transform(
            first1, last1, std::forward<InputIt2>( first2 ), std::forward<OutputIt>( d_first ),
            std::forward<BinaryOp>( binary_op ) );
}

} // namespace cuda

} // namespace Backend

} // namespace Engine

#endif
