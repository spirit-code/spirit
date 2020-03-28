#pragma once
#ifndef BACKEND_PAR_H
#define BACKEND_PAR_H

#include <engine/Vectormath_Defines.hpp>

#include <tuple>

#ifdef SPIRIT_USE_CUDA
    #include <cub/cub.cuh>
    #define SPIRIT_LAMBDA __device__
#else
    #define SPIRIT_LAMBDA
#endif

namespace Engine
{
namespace Backend
{
namespace par
{

#ifdef SPIRIT_USE_CUDA

    // f(i) for all i
    template<typename F>
    __global__
    void cu_apply(int N, F f)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
            f(idx);
    }
    // f(i) for all i
    template<typename F>
    void apply(int N, F f)
    {
        cu_apply<<<(N+1023)/1024, 1024>>>(N, f);
        CU_CHECK_AND_SYNC();
    }

    // vf_to[i] = f(args[i]...) for all i
    template<typename A, typename Lambda, typename... args>
    __global__
    void cu_assign_lambda(int N, A * vf_to, Lambda lambda, const args *... arg_fields)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
            vf_to[idx] = lambda(arg_fields[idx]...);
    }
    // field[i] = f(args[i]...) for all i
    template<typename A, typename Lambda, typename... args>
    void assign(field<A> & vf_to, Lambda lambda, const field<args> &... vf_args)
    {
        int N = vf_to.size();
        cu_assign_lambda<<<(N+1023)/1024, 1024>>>(N, vf_to.data(), lambda, vf_args.data()...);
        CU_CHECK_AND_SYNC();
    }

    // result = sum_i f(args[i]...)
    template<typename Lambda, typename... args> inline
    scalar pointer_reduce(int N, const Lambda & lambda, const args *... vf_args)
    {
        // TODO: remove the reliance on a temporary scalar field (maybe thrust::dot with generalized operations)
        static scalarfield sf(N, 0);
        if(sf.size() != N)
            sf.resize(N);

        auto s  = sf.data();
        apply( N, [s, vf_args...] SPIRIT_LAMBDA (int idx) { s[idx] = f(vf_args[idx]...); } );

        static scalarfield ret(1, 0);
        // Determine temporary storage size and allocate
        void * d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size());
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, sf.data(), ret.data(), sf.size());
        CU_CHECK_AND_SYNC();
        cudaFree(d_temp_storage);
        return ret[0];
    }
    // result = sum_i f(args[i]...)
    template<typename Lambda, typename... args> inline
    scalar reduce(const Lambda & lambda, const field<args> &... vf_args)
    {
        auto N = std::get<0>(std::tuple<const field<args> &...>(vf_args...)).size();
        // We take this umweg so that `.data()` won't be called for every element of each field
        return pointer_reduce(N, lambda, vf_args.data()...);
    }

#else

    // f(i) for all i
    template<typename F>
    void apply(int N, const F & f)
    {
        #pragma omp parallel for
        for(unsigned int idx = 0; idx < N; ++idx)
            f(idx);
    }

    // vf_to[i] = f(args[i]...) for all i
    template<typename A, typename Lambda, typename... args> inline
    void pointer_assign(int N, A * vf_to, Lambda lambda, const args *... vf_args)
    {
        #pragma omp parallel for
        for(unsigned int idx = 0; idx < N; ++idx)
            vf_to[idx] = lambda(vf_args[idx]...);
    }
    // vf_to[i] = f(args[i]...) for all i
    template<typename A, typename Lambda, typename... args> inline
    void assign(field<A> & vf_to, Lambda lambda, const field<args> &... vf_args)
    {
        auto N = vf_to.size();
        // We take this umweg so that `.data()` won't be called for every element of each field
        pointer_assign(N, vf_to.data(), lambda, vf_args.data()...);
    }


    // result = sum_i f(args[i]...)
    template<typename Lambda, typename... args> inline
    scalar pointer_reduce(int N, const Lambda & lambda, const args *... vf_args)
    {
        scalar res = 0;

        #pragma omp parallel for reduction(+:res)
        for(unsigned int idx = 0; idx < N; ++idx)
            res += lambda(vf_args[idx]...);

        return res;
    }
    // result = sum_i f(args[i]...)
    template<typename Lambda, typename... args> inline
    scalar reduce(const Lambda & lambda, const field<args> &... vf_args)
    {
        auto N = std::get<0>(std::tuple<const field<args> &...>(vf_args...)).size();
        // We take this umweg so that `.data()` won't be called for every element of each field
        return pointer_reduce(N, lambda, vf_args.data()...);
    }

#endif

} // namespace par
} // namespace Backend
} // namespace Engine

#endif