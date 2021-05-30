#include <Spirit/Version.h>
#include <utility/Version.hpp>

const int Spirit_Version_Major() noexcept
{
    return Utility::version_major;
}

const int Spirit_Version_Minor() noexcept
{
    return Utility::version_minor;
}

const int Spirit_Version_Patch() noexcept
{
    return Utility::version_patch;
}

const char * Spirit_Version() noexcept
{
    return Utility::version.c_str();
}

const char * Spirit_Version_Revision() noexcept
{
    return Utility::version_revision.c_str();
}

const char * Spirit_Version_Full() noexcept
{
    return Utility::version_full.c_str();
}

const char * Spirit_Compiler() noexcept
{
    return Utility::compiler.c_str();
}

const char * Spirit_Compiler_Version() noexcept
{
    return Utility::compiler_version.c_str();
}

const char * Spirit_Compiler_Full() noexcept
{
    return Utility::compiler_full.c_str();
}

const char * Spirit_Scalar_Type() noexcept
{
    return Utility::scalartype.c_str();
}

const char * Spirit_Pinning() noexcept
{
    return Utility::pinning.c_str();
}

const char * Spirit_Defects() noexcept
{
    return Utility::defects.c_str();
}

const char * Spirit_FFTW() noexcept
{
    return Utility::fftw.c_str();
}

const char * Spirit_Cuda() noexcept
{
    return Utility::cuda.c_str();
}

const char * Spirit_OpenMP() noexcept
{
    return Utility::openmp.c_str();
}

const char * Spirit_Threads() noexcept
{
    return Utility::threads.c_str();
}