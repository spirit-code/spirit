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
    template<typename Lambda>
    __global__
    void cu_apply(int N, Lambda f)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
            f(idx);
    }
    // f(i) for all i
    template<typename Lambda>
    void apply(int N, Lambda f)
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


    template<typename Lambda>
    scalar reduce(int N, const Lambda f)
    {
        static scalarfield sf(N, 0);

        if(sf.size() != N)
            sf.resize(N);

        auto s  = sf.data();
        apply( N, [f, s] SPIRIT_LAMBDA (int idx) { s[idx] = f(idx); } );

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

    template<typename A, typename Lambda>
    scalar reduce(int N, const Lambda f, const field<A> & vf1)
    {
        // TODO: remove the reliance on a temporary scalar field (maybe thrust::dot with generalized operations)
        // We also use this workaround for a single field as argument, because cub does not support non-commutative reduction operations

        int n = vf1.size();
        static scalarfield sf(n, 0);

        if(sf.size() != vf1.size())
            sf.resize(vf1.size());

        auto s  = sf.data();
        auto v1 = vf1.data();
        apply( n, [f, s, v1] SPIRIT_LAMBDA (int idx) { s[idx] = f(v1[idx]); } );

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

    template<typename A, typename B, typename F>
    scalar reduce(int N, const F f, const field<A> & vf1, const field<B> & vf2)
    {
        // TODO: remove the reliance on a temporary scalar field (maybe thrust::dot with generalized operations)
        int n = vf1.size();
        static scalarfield sf(n, 0);

        if(sf.size() != vf1.size())
            sf.resize(vf1.size());

        auto s  = sf.data();
        auto v1 = vf1.data();
        auto v2 = vf2.data();
        apply( n, [f, s, v1, v2] SPIRIT_LAMBDA (int idx) { s[idx] = f(v1[idx], v2[idx]); } );

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
    scalar reduce(int N, const Lambda & lambda, const field<args> &... vf_args)
    {
        // We take this umweg so that `.data()` won't be called for every element of each field
        return pointer_reduce(N, lambda, vf_args.data()...);
    }

#endif

} // namespace par
} // namespace Backend
} // namespace Engine

#endif