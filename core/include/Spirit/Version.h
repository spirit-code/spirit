#pragma once
#ifndef VERSION_H
#define VERSION_H
#include "DLL_Define_Export.h"

DLLEXPORT const int Spirit_Version_Major() noexcept;
DLLEXPORT const int Spirit_Version_Minor() noexcept;
DLLEXPORT const int Spirit_Version_Patch() noexcept;

DLLEXPORT const char * Spirit_Version() noexcept;
DLLEXPORT const char * Spirit_Version_Revision() noexcept;
DLLEXPORT const char * Spirit_Version_Full() noexcept;

DLLEXPORT const char * Spirit_Compiler() noexcept;
DLLEXPORT const char * Spirit_Compiler_Version() noexcept;
DLLEXPORT const char * Spirit_Compiler_Full() noexcept;

#include "DLL_Undefine_Export.h"
#endif