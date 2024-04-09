#pragma once
#ifndef SPIRIT_CORE_ENGINE_BACKEND_PAR_HPP
#define SPIRIT_CORE_ENGINE_BACKEND_PAR_HPP

#include <engine/backend/Counting_Iterator.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/backend/types.hpp>
#include <engine/backend/algorithms_openmp.hpp>
#include <engine/backend/algorithms_cuda.cuh>

#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

#ifdef SPIRIT_USE_STDPAR
#include <execution>
#define SPIRIT_CPU_PAR std::execution::par,
#elseifdef SPIRIT_USE_OPENMP

#define SPIRIT_CPU_PAR ::execution::par,
#else
#define SPIRIT_CPU_PAR
#endif

#ifndef SPIRIT_USE_CUDA
#define SPIRIT_PAR SPIRIT_CPU_PAR
#define SPIRIT_LAMBDA
#else
#define SPIRIT_PAR
#define SPIRIT_LAMBDA __device__
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
using std::transform;
using std::transform_reduce;

} // namespace cpu

#ifndef SPIRIT_USE_CUDA
using namespace cpu;
#else
using namespace cuda;
#endif

} // namespace Backend

} // namespace Engine
#endif
