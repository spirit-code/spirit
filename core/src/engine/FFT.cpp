#ifndef SPIRIT_USE_CUDA

#include <engine/FFT.hpp>

#include <iostream>
#include <vector>

namespace Engine
{
namespace FFT
{

//=== Functions for FFTW backend ===
#ifdef SPIRIT_USE_FFTW

// Dont need the single transforms because FFTW can do real batch transforms
void Four_3D( const FFT_cfg & cfg, FFT_real_type * in, FFT_cpx_type * out )
{
    std::cerr << "NOT IMPLEMENTED FOR FFTW!" << std::endl;
}
void iFour_3D( const FFT_cfg & cfg, FFT_cpx_type * in, FFT_real_type * out )
{
    std::cerr << "NOT IMPLEMENTED FOR FFTW!" << std::endl;
}

void batch_Four_3D( FFT_Plan & plan )
{
    FFTW_EXECUTE( plan.cfg );
}

void batch_iFour_3D( FFT_Plan & plan )
{
    FFTW_EXECUTE( plan.cfg );
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
        size *= k;

    int idist = 1, odist = 1;

    if( this->inverse == false )
        this->cfg = FFTW_PLAN_MANY_DFT_R2C(
            rank, n, n_transforms, this->real_ptr.data(), inembed, istride, idist,
            reinterpret_cast<FFTW_COMPLEX *>( this->cpx_ptr.data() ), onembed, ostride, odist, FFTW_MEASURE );
    else
        this->cfg = FFTW_PLAN_MANY_DFT_C2R(
            rank, n, n_transforms, reinterpret_cast<FFTW_COMPLEX *>( this->cpx_ptr.data() ), inembed, istride, idist,
            this->real_ptr.data(), onembed, ostride, odist, FFTW_MEASURE );
}

void FFT_Plan::Free_Configuration()
{
    FFTW_DESTROY_PLAN( this->cfg );
}

void FFT_Plan::Clean()
{
    this->cpx_ptr  = field<FFT_cpx_type>();
    this->real_ptr = field<FFT_real_type>();
    Free_Configuration();
}
#endif // end fftw_backend

//=== Functions for kissFFT backend ===
#ifdef SPIRIT_USE_KISSFFT
void Four_3D( const FFT_cfg & cfg, FFT_real_type * in, FFT_cpx_type * out )
{
    kiss_fftndr( cfg, in, out );
}

void iFour_3D( const FFT_cfg & cfg, FFT_cpx_type * in, FFT_real_type * out )
{
    kiss_fftndri( cfg, in, out );
}

void batch_Four_3D( FFT_Plan & plan )
{
    int number = plan.n_transforms;
    int size   = 1;
    for( auto k : plan.dims )
        size *= k;
    const auto & in  = plan.real_ptr.data();
    const auto & out = plan.cpx_ptr.data();

    // just for testing
    // const auto& in = plan.real_ptr;
    // const auto& out = plan.cpx_ptr;

    for( int dir = 0; dir < number; ++dir )
        Engine::FFT::Four_3D( plan.cfg, in + dir * size, out + dir * size );
}

// same as above but iFFT
void batch_iFour_3D( FFT_Plan & plan )
{
    int number = plan.n_transforms;
    int size   = 1;
    for( auto k : plan.dims )
        size *= k;

    const auto & in  = plan.cpx_ptr.data();
    const auto & out = plan.real_ptr.data();

    for( int dir = 0; dir < number; ++dir )
        Engine::FFT::iFour_3D( plan.cfg, in + dir * size, out + dir * size );
}

void FFT_Plan::Create_Configuration()
{
    this->cfg = kiss_fftndr_alloc( this->dims.data(), this->dims.size(), this->inverse, NULL, NULL );
}

void FFT_Plan::Free_Configuration()
{
    free( this->cfg );
}
#endif // end kiss_fft backend

} // namespace FFT
} // namespace Engine

#endif