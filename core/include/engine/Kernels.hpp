#pragma once
#ifndef KERNELS_HPP
#define KERNELS_HPP
#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>

namespace Engine {
    namespace Kernels {
        #ifndef SPIRIT_USE_CUDA

        template<typename Func>
        scalar reduce(int end, Func f)
        {
            scalar res=0;
            #pragma omp parallel for reduce(+:res)
            for(unsigned int idx = 0; idx < end; ++idx)
            {
                res += f(idx);
            }
            return res;
        }

        template<typename A, typename Func>
        scalar set(field<A> & vf, Func f)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < vf.size(); ++idx)
            {
                vf[idx] = f(idx);
            }
        }

        template<typename A, typename Func>
        scalar add(field<A> & vf, Func f)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < vf.size(); ++idx)
            {
                vf[idx] += f(idx);
            }
        }

        template<typename A, typename Func>
        scalar apply(field<A> & vf, Func f)
        {
            #pragma omp parallel for
            for(unsigned int idx = 0; idx < vf.size(); ++idx)
            {
                f(idx, vf[idx]);
            }
        }

        // // Maybe a nice idea ..
        // template<typename A, typename Func, typename bounds ...>
        // scalar apply(field<A> & vf, Func f)
        // {
        //     for(int i=0; i<bounds; i++)
        //         apply(field<A> & vf, f, bounds)
        //     }
        // }

        #else

        #endif
    }
}

#endif