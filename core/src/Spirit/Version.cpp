#include <Spirit/Version.h>
#include <utility/Version.hpp>

#ifdef SPIRIT_USE_OPENMP
#include <omp.h>
#endif

namespace Version = Spirit::Utility::Version;

int Spirit_Version_Major() noexcept
{
    return Version::major;
}

int Spirit_Version_Minor() noexcept
{
    return Version::minor;
}

int Spirit_Version_Patch() noexcept
{
    return Version::patch;
}

const char * Spirit_Version() noexcept
{
    return Version::version.c_str();
}

const char * Spirit_Version_Revision() noexcept
{
    return Version::revision.c_str();
}

const char * Spirit_Version_Full() noexcept
{
    return Version::full.c_str();
}

const char * Spirit_Compiler() noexcept
{
    return Version::compiler.c_str();
}

const char * Spirit_Compiler_Version() noexcept
{
    return Version::compiler_version.c_str();
}

const char * Spirit_Compiler_Full() noexcept
{
    return Version::compiler_full.c_str();
}

const char * Spirit_Scalar_Type() noexcept
{
    return Version::scalartype.c_str();
}

const char * Spirit_Pinning() noexcept
{
    return Version::pinning.c_str();
}

const char * Spirit_Defects() noexcept
{
    return Version::defects.c_str();
}

const char * Spirit_FFTW() noexcept
{
    return Version::fftw.c_str();
}

const char * Spirit_Cuda() noexcept
{
    return Version::cuda.c_str();
}

const char * Spirit_OpenMP() noexcept
{
    return Version::openmp.c_str();
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
    return Version::threads.c_str();
}