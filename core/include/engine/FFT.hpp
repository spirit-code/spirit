#pragma once
#ifndef FFT_H
#define FFT_H
#include "Eigen/Core"
#include <data/Geometry.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <iostream>
#include <complex>

#ifdef SPIRIT_USE_OPENMP
#include <omp.h>
#endif

#ifdef SPIRIT_USE_KISSFFT
#include "kiss_fft.h"
#include "kiss_fftndr.h"
#endif

#ifdef SPIRIT_USE_FFTW
#include <array>
#include <fftw3.h>
#endif

#ifdef SPIRIT_USE_CUDA
#include <cufft.h>
#endif

namespace Engine 
{
    namespace FFT
    {
        #ifdef SPIRIT_USE_KISSFFT
        using FFT_real_type = kiss_fft_scalar;
        using FFT_cpx_type = kiss_fft_cpx;
        using FFT_cfg = kiss_fftndr_cfg;

        //scalar product of two complex vectors
        inline FFT_cpx_type mult3D(FFT_cpx_type & d1, FFT_cpx_type & d2, FFT_cpx_type & d3, FFT_cpx_type & s1, FFT_cpx_type & s2, FFT_cpx_type & s3)
        {
            FFT_cpx_type res;
            res.r = d1.r * s1.r + d2.r * s2.r + d3.r * s3.r - d1.i * s1.i - d2.i * s2.i - d3.i * s3.i;
            res.i = d1.r * s1.i + d2.r * s2.i + d3.r * s3.i + d1.i * s1.r + d2.i * s2.r + d3.i * s3.r;
            return res;
        }

        inline void addTo(FFT_cpx_type & a, const FFT_cpx_type & b, bool overwrite)
        {
            if(overwrite)
            {
                a.r = b.r;
                a.i = b.i;
            } else {
            a.r += b.r;
            a.i += b.i;
        }
        }
        #endif

        #ifdef SPIRIT_USE_FFTW
        using FFT_real_type = scalar;
        using FFT_cpx_type = std::array<scalar, 2>;
        using FFT_cfg = fftw_plan;

        inline FFT_cpx_type mult3D(FFT_cpx_type & d1, FFT_cpx_type & d2, FFT_cpx_type & d3, FFT_cpx_type & s1, FFT_cpx_type & s2, FFT_cpx_type & s3)
        {
            FFT_cpx_type res;
            res[0] = d1[0] * s1[0] + d2[0] * s2[0] + d3[0] * s3[0] - d1[1] * s1[1] - d2[1] * s2[1] - d3[1] * s3[1];
            res[1] = d1[0] * s1[1] + d2[0] * s2[1] + d3[0] * s3[1] + d1[1] * s1[0] + d2[1] * s2[0] + d3[1] * s3[0];
            return res;     
        }

        inline void addTo(FFT_cpx_type & a, const FFT_cpx_type & b, bool overwrite)
        {
            if(overwrite)
            {
                a[0] = b[0];
                a[1] = b[1];
            } else {
                a[0] += b[0];
                a[1] += b[1];
            }
        }
        #endif

        #ifdef SPIRIT_USE_CUDA
        //these are single precision types!
        using FFT_real_type = cufftReal;
        using FFT_cpx_type = cufftComplex;
        using FFT_cfg = cufftHandle;

        //scalar product of two complex vectors
        inline __device__ FFT_cpx_type mult3D(FFT_cpx_type & d1, FFT_cpx_type & d2, FFT_cpx_type & d3, FFT_cpx_type & s1, FFT_cpx_type & s2, FFT_cpx_type & s3)
        {
            FFT_cpx_type res;
            res.x = d1.x * s1.x + d2.x * s2.x + d3.x * s3.x - d1.y * s1.y - d2.y * s2.y - d3.y * s3.y;
            res.y = d1.x * s1.y + d2.x * s2.y + d3.x * s3.y + d1.y * s1.x + d2.y * s2.x + d3.y * s3.x;
            return res;   
        }

        inline __device__ void addTo(FFT_cpx_type & a, const FFT_cpx_type & b, bool overwrite = false)
        {
            if(overwrite)
            {
                a.x = b.x;
                a.y = b.y;
            } else {
                a.x += b.x;
                a.y += b.y;
            }
        }
        #endif

        inline void get_strides(field<int*> & strides, const field<int> & maxVal)
        {
            strides.resize(maxVal.size());
            *(strides[0]) = 1;
            for(int i = 1; i < maxVal.size(); i++)
            {
                *(strides[i]) = *(strides[i-1]) * maxVal[i-1];
            }
        }

        struct StrideContainer
        {
            int comp;
            int basis;
            int a;
            int b;
            int c;
        };

        struct FFT_Plan
        {
            std::vector<int> dims;
            bool inverse;
            int howmany;

            field<FFT_cpx_type> cpx_ptr;
            field<FFT_real_type> real_ptr;

            std::string name;

            void CreateConfiguration();
            FFT_cfg cfg;

            // FFT_Plan(std::string name);
            ~FFT_Plan();
        };

        inline void FFT_Init()
        {
            #if defined SPIRIT_USE_FFTW && defined SPIRIT_USE_OPENMP
                fftw_init_threads();
                fftw_plan_with_nthreads(omp_get_max_threads());
            #endif
        }

        void Four_3D(const FFT_Plan & plan, field<FFT_real_type> & in, field<FFT_cpx_type> & out);
        void iFour_3D(const FFT_Plan & plan, field<FFT_cpx_type> & in, field<FFT_real_type> & out);
        void batch_Four_3D(FFT_Plan & plan);
        void batch_iFour_3D(FFT_Plan & plan);
    }
}
#endif