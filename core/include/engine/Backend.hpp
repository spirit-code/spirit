#pragma once
#ifndef SPIRIT_CORE_ENGINE_BACKEND_HPP
#define SPIRIT_CORE_ENGINE_BACKEND_HPP

#include <engine/Vectormath_Defines.hpp>
#include <engine/backend/Counting_Iterator.hpp>
#include <engine/backend/Functor.hpp>
#include <engine/backend/Transform_Iterator.hpp>
#include <engine/backend/Zip_Iterator.hpp>
#include <engine/backend/algorithms_cuda.cuh>
#include <engine/backend/algorithms_openmp.hpp>
#include <engine/backend/types.hpp>

#include <algorithm>
#include <numeric>

#if defined( SPIRIT_USE_STDPAR )
#include <execution>
#define SPIRIT_CPU_PAR std::execution::par,
#elif defined( SPIRIT_USE_OPENMP )
#define SPIRIT_CPU_PAR ::execution::par,
#else
#define SPIRIT_CPU_PAR
#endif

#if !defined( SPIRIT_USE_CUDA )
#define SPIRIT_PAR SPIRIT_CPU_PAR
#define SPIRIT_LAMBDA
#else
#define SPIRIT_PAR ::execution::cuda::par,
#define SPIRIT_LAMBDA SPIRIT_HOSTDEVICE
#endif

namespace Engine
{

namespace Backend
{

namespace cpu
{

using std::copy;
using std::copy_n;
using std::fill;
using std::fill_n;
using std::for_each;
using std::for_each_n;
using std::reduce;
using std::transform;
using std::transform_reduce;

} // namespace cpu

#if !defined( SPIRIT_USE_CUDA )
using namespace Backend::cpu;
#else
using namespace Backend::cuda;
#endif

} // namespace Backend

} // namespace Engine
#endif
