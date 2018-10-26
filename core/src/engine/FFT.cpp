#ifndef SPIRIT_USE_CUDA
#include "FFT.hpp"
#include <iostream>
#include <vector>

namespace Engine 
{
    namespace FFT
    {
        //=== Functions for FFTW backend ===
        #ifdef SPIRIT_USE_FFTW

        //Dont need the single transforms because FFTW can do real batch transforms
        void Four_3D(const FFT_cfg & cfg, FFT_real_type * in, FFT_cpx_type * out)
        {
            std::cerr << "NOT IMPLEMENTED FOR FFTW!" << std::endl;
        }
        void iFour_3D(const FFT_cfg & cfg, FFT_cpx_type * in, FFT_real_type * out)
        {
            std::cerr << "NOT IMPLEMENTED FOR FFTW!" << std::endl;   
        }

        void batch_Four_3D(FFT_Plan & plan)
        {
            FFTW_EXECUTE(plan.cfg);
            // fftw_execute(plan.cfg);
        }

        void batch_iFour_3D(FFT_Plan & plan)
        {
            FFTW_EXECUTE(plan.cfg);
            // fftw_execute(plan.cfg);
        }

        void FFT_Plan::Create_Configuration()
        {
            int rank = this->dims.size();
            int *n = this->dims.data();
            int howmany = this->howmany;
            int istride = howmany, ostride = howmany;
            int *inembed = n, *onembed = n;
            
            int size = 1;
            for(auto k : dims)
                size *= k;

            int idist = 1, odist = 1;

            if(this->inverse == false)
                // this->cfg = fftw_plan_many_dft_r2c(rank, n, howmany, this->real_ptr.data(), inembed, istride, idist, reinterpret_cast<fftw_complex*>(this->cpx_ptr.data()), onembed, ostride, odist, FFTW_MEASURE);
                this->cfg = FFTW_PLAN_MANY_DFT_R2C(rank, n, howmany, this->real_ptr.data(), inembed, istride, idist, reinterpret_cast<FFTW_COMPLEX*>(this->cpx_ptr.data()), onembed, ostride, odist, FFTW_MEASURE);
            else
                this->cfg = FFTW_PLAN_MANY_DFT_C2R(rank, n, howmany, reinterpret_cast<FFTW_COMPLEX*>(this->cpx_ptr.data()), inembed, istride, idist, this->real_ptr.data(), onembed, ostride, odist, FFTW_MEASURE);
            this->freeable = true;
        }

        void FFT_Plan::Free_Configuration()
        {
            if(freeable)
            {
                // fftw_destroy_plan(this->cfg);
                FFTW_DESTROY_PLAN(this->cfg);
                this->freeable = false;
            }
        }

        void FFT_Plan::Clean()
        {
            this->cpx_ptr = field<FFT_cpx_type>();
            this->real_ptr = field<FFT_real_type>();
            Free_Configuration();
        }

        // FFT_Plan::FFT_Plan(std::string name) : name(name) {

        //     std::cerr << "Calling constructor " << name << std::endl;
        // }

        FFT_Plan::~FFT_Plan()
        {
            // std::cerr << "Calling Destructor " << name << std::endl;
        }

        #endif


        //=== Functions for kissFFT backend ===
        #ifdef SPIRIT_USE_KISSFFT
        void Four_3D(const FFT_cfg & cfg, FFT_real_type * in, FFT_cpx_type * out)
        {
            kiss_fftndr(cfg, in, out);
        }

        void iFour_3D(const FFT_cfg & cfg, FFT_cpx_type * in, FFT_real_type * out)
        {
            kiss_fftndri(cfg, in, out);     
        }

        void batch_Four_3D(FFT_Plan & plan)
        {
            // std::cerr << "Calling batch Four" << std::endl;
            int number = plan.howmany;
            int size = 1;
            for(auto k : plan.dims)
                size *= k;
            const auto& in = plan.real_ptr.data();
            const auto& out = plan.cpx_ptr.data();

            //just for testing
            // const auto& in = plan.real_ptr;
            // const auto& out = plan.cpx_ptr;

            for(int dir = 0; dir < number; ++dir)
                Engine::FFT::Four_3D(plan.cfg, in + dir * size, out + dir * size);
            
        }

        //same as above but iFFT
        void batch_iFour_3D(FFT_Plan & plan)
        {
            int number = plan.howmany;
            int size = 1;
            for(auto k : plan.dims)
                size *= k;

            const auto& in = plan.cpx_ptr.data();
            const auto& out = plan.real_ptr.data();

            for(int dir = 0; dir < number; ++dir)
                Engine::FFT::iFour_3D(plan.cfg, in + dir * size, out + dir * size);
        }

        void FFT_Plan::Create_Configuration()
        {
            this->cfg = kiss_fftndr_alloc(this->dims.data(), this->dims.size(), this->inverse, NULL, NULL);
            this->freeable = true;
        }

        void FFT_Plan::Free_Configuration()
        {
            if(freeable)
            {
                free(this->cfg);
                this->freeable = false;
            }
        }

        void FFT_Plan::Clean()
        {
            this->cpx_ptr = field<FFT_cpx_type>();
            this->real_ptr = field<FFT_real_type>();
            Free_Configuration();
        }

        FFT_Plan::~FFT_Plan()
        {
            std::cerr << "Calling Destructor " << name << std::endl;
        }
        #endif
    }
}
#endif