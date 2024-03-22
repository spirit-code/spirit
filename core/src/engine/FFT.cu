#ifdef SPIRIT_USE_CUDA

#include <engine/FFT.hpp>

#include <cufft.h>

namespace Engine
{
namespace FFT
{

// Dont need the single transforms because cuFFT can do real batch transforms
void Four_3D( FFT_cfg cfg, FFT_real_type * in, FFT_cpx_type * out )
{
    std::cerr << "NOT IMPLEMENTED FOR cuFFT" << std::endl;
}

void iFour_3D( FFT_cfg cfg, FFT_cpx_type * in, FFT_real_type * out )
{
    std::cerr << "NOT IMPLEMENTED FOR cuFFT" << std::endl;
}

void batch_Four_3D( FFT_Plan & plan )
{
    auto res = cufftExecR2C( plan.cfg, raw_pointer_cast( plan.real_ptr.data() ), raw_pointer_cast( plan.cpx_ptr.data() ) );
    if( res != CUFFT_SUCCESS )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format( "cufftExecR2C failed with error: {}", static_cast<int>( res ) ) );
    }
    cudaDeviceSynchronize();
}

void batch_iFour_3D( FFT_Plan & plan )
{
    auto res = cufftExecC2R( plan.cfg, raw_pointer_cast( plan.cpx_ptr.data() ), raw_pointer_cast( plan.real_ptr.data() ) );
    if( res != CUFFT_SUCCESS )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format( "cufftExecC2R failed with error: {}", static_cast<int>( res ) ) );
    }
    cudaDeviceSynchronize();
}

void FFT_Plan::Create_Configuration()
{
    int rank         = this->dims.size();
    int * n          = this->dims.data();
    int n_transforms = this->n_transforms;
    int istride = n_transforms, ostride = n_transforms;
    int *inembed = n, *onembed = n;

    int size = 1;
    for( auto k : dims )
    {
        size *= k;
    }
    int idist = 1, odist = 1;

    if( this->inverse == false )
    {
        auto res = cufftPlanMany(
            &this->cfg, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, n_transforms );
        if( res != CUFFT_SUCCESS )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format( "cufftPlanMany failed with error: {}", static_cast<int>( res ) ) );
        }
    }
    else
    {
        auto res = cufftPlanMany(
            &this->cfg, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2R, n_transforms );
        if( res != CUFFT_SUCCESS )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format( "cufftPlanMany failed with error: {}", static_cast<int>( res ) ) );
        }
    }
}

void FFT_Plan::Free_Configuration()
{
    auto res = cufftDestroy( this->cfg );
    if( res != CUFFT_SUCCESS )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format( "cufftDestroy( {} ) failed with error: {}", this->cfg, static_cast<int>( res ) ) );
    }
}

} // namespace FFT
} // namespace Engine

#endif
