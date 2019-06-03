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