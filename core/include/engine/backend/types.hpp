#pragma once

#include <engine/Vectormath_Defines.hpp>
#include <engine/backend/Counting_Iterator.hpp>

#include <functional>
#include <optional>
#include <tuple>
#include <vector>

#ifdef SPIRIT_USE_STDPAR
#include <execution>

#elif defined( SPIRIT_USE_OPENMP )

namespace execution
{

struct par_t
{
    constexpr par_t() noexcept = default;
};

bool constexpr operator==( const par_t &, const par_t & )
{
    return true;
};
bool constexpr operator!=( const par_t &, const par_t & )
{
    return false;
};

static constexpr par_t par = par_t();

} // namespace execution

#endif

#ifdef SPIRIT_USE_CUDA
#include <thrust/execution_policy.h>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/functional>

namespace execution
{

namespace cuda
{

struct par_t
{
    constexpr par_t() noexcept = default;
};

bool constexpr operator==( const par_t &, const par_t & )
{
    return true;
};
bool constexpr operator!=( const par_t &, const par_t & )
{
    return false;
};

static constexpr par_t par = par_t();

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

using std::divides;
using std::minus;
using std::modulus;
using std::multiplies;
using std::negate;
using std::plus;

} // namespace cpu

#ifdef SPIRIT_USE_CUDA

namespace cuda
{

using ::cuda::std::optional;
template<typename T>
using vector = field<T>;

using ::cuda::std::apply;
using ::cuda::std::get;
using ::cuda::std::make_tuple;
using ::cuda::std::tuple;

using ::cuda::std::divides;
using ::cuda::std::minus;
using ::cuda::std::modulus;
using ::cuda::std::multiplies;
using ::cuda::std::negate;
using ::cuda::std::plus;

} // namespace cuda

using namespace Backend::cuda;

#else
using namespace Backend::cpu;
#endif

} // namespace Backend

} // namespace Engine
