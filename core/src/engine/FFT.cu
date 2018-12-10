#ifdef SPIRIT_USE_CUDA
#include "FFT.hpp"
#include <cufft.h>
namespace Engine
{
    namespace FFT
    {
        //Dont need the single transforms because cuFFT can do real batch transforms
        void Four_3D(FFT_cfg cfg, FFT_real_type * in, FFT_cpx_type * out)
        {
            std::cerr << "NOT IMPLEMENTED FOR cuFFT" << std::endl;
        }
        void iFour_3D(FFT_cfg cfg, FFT_cpx_type * in, FFT_real_type * out)
        {
            std::cerr << "NOT IMPLEMENTED FOR cuFFT" << std::endl;   
        }

        void batch_Four_3D(FFT_Plan & plan)
        {
            cufftExecR2C(plan.cfg, plan.real_ptr.data(), plan.cpx_ptr.data());
            cudaDeviceSynchronize();
        }

        void batch_iFour_3D(FFT_Plan & plan)
        {
            cufftExecC2R(plan.cfg, plan.cpx_ptr.data(), plan.real_ptr.data());
            cudaDeviceSynchronize();
        }

        void FFT_Plan::Create_Configuration()
        {
            int rank = this->dims.size();
            int *n = this->dims.data();
            int n_transforms = this->n_transforms;
            int istride = n_transforms, ostride = n_transforms;
            int *inembed = n, *onembed = n;
            
            int size = 1;
            for(auto k : dims)
            {
                size *= k;
            }
            int idist = 1, odist = 1;

            if(this->inverse == false)
            {
                cufftPlanMany(&this->cfg, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, n_transforms);
            } else 
            {
                cufftPlanMany(&this->cfg, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, n_transforms);
            }
        }

        void FFT_Plan::Free_Configuration()
        {
            cufftDestroy(this->cfg);
        }
    }
}
#endif