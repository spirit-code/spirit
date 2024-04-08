#pragma once

#include <engine/Counting_Iterator.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <functional>
#include <optional>
#include <tuple>
#include <vector>

#ifdef SPIRIT_USE_STDPAR
#include <execution>

#elseifdef SPIRIT_USE_OPENMP

namespace execution
{

struct Par
{
    constexpr Par() noexcept = default;
};

bool constexpr operator==( const Par & first, const Par & second )
{
    return true;
};
bool constexpr operator!=( const Par & first, const Par & second )
{
    return false;
};

static constexpr Par par = Par();

} // namespace execution

#endif

#ifdef SPIRIT_USE_CUDA
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>

#include <cuda/std/tuple>

namespace execution
{

namespace cuda
{

struct Par
{
    constexpr Par() noexcept = default;
};

bool constexpr operator==( const Par & first, const Par & second )
{
    return true;
};
bool constexpr operator!=( const Par & first, const Par & second )
{
    return false;
};

static constexpr Par par = Par();

} // namespace cuda

} // namespace execution

#endif

namespace Engine
{

namespace Backend
{

namespace cpu
{

using std::optional;
using std::vector;

using std::apply;
using std::get;
using std::make_tuple;
using std::tuple;

using std::plus;


} // namespace cpu

#ifdef SPIRIT_USE_CUDA
using thrust::optional;
template<typename T>
using vector = field<T>;

using cuda::std::apply;
using cuda::std::get;
using cuda::std::make_tuple;
using cuda::std::tuple;

using thrust::plus;

#else
using namespace cpu;
#endif

} // namespace Backend

} // namespace Engine
