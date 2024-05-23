#include <Spirit/Version.h>
#include <utility/Version.hpp>

#ifdef SPIRIT_USE_OPENMP
#include <omp.h>
#endif

int Spirit_Version_Major() noexcept
{
    return Utility::version_major;
}

int Spirit_Version_Minor() noexcept
{
    return Utility::version_minor;
}

int Spirit_Version_Patch() noexcept
{
    return Utility::version_patch;
}

const char * Spirit_Version() noexcept
{
    return Utility::version.data();
}

const char * Spirit_Version_Revision() noexcept
{
    return Utility::version_revision.data();
}

const char * Spirit_Version_Full() noexcept
{
    return Utility::version_full.data();
}

const char * Spirit_Compiler() noexcept
{
    return Utility::compiler.data();
}

const char * Spirit_Compiler_Version() noexcept
{
    return Utility::compiler_version.data();
}

const char * Spirit_Compiler_Full() noexcept
{
    return Utility::compiler_full.data();
}

const char * Spirit_Scalar_Type() noexcept
{
    return Utility::scalartype.data();
}

const char * Spirit_Pinning() noexcept
{
    return Utility::pinning.data();
}

const char * Spirit_Defects() noexcept
{
    return Utility::defects.data();
}

const char * Spirit_FFTW() noexcept
{
    return Utility::fftw.data();
}

const char * Spirit_Cuda() noexcept
{
    return Utility::cuda.data();
}

const char * Spirit_OpenMP() noexcept
{
    return Utility::openmp.data();
}

int Spirit_OpenMP_Get_Num_Threads() noexcept
{
#ifdef SPIRIT_USE_OPENMP
    int n_threads{ 1 };
#pragma omp parallel
    {
#pragma omp single
        n_threads = omp_get_num_threads();
    }
    return n_threads;
#else
    return 1;
#endif
}

const char * Spirit_Threads() noexcept
{
    return Utility::threads.data();
}
