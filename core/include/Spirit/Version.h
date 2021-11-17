#pragma once
#ifndef SPIRIT_CORE_VERSION_H
#define SPIRIT_CORE_VERSION_H
#include "DLL_Define_Export.h"

PREFIX const int Spirit_Version_Major() SUFFIX;
PREFIX const int Spirit_Version_Minor() SUFFIX;
PREFIX const int Spirit_Version_Patch() SUFFIX;

PREFIX const char * Spirit_Version() SUFFIX;
PREFIX const char * Spirit_Version_Revision() SUFFIX;
PREFIX const char * Spirit_Version_Full() SUFFIX;

PREFIX const char * Spirit_Compiler() SUFFIX;
PREFIX const char * Spirit_Compiler_Version() SUFFIX;
PREFIX const char * Spirit_Compiler_Full() SUFFIX;

PREFIX const char * Spirit_Scalar_Type() SUFFIX;

PREFIX const char * Spirit_Defects() SUFFIX;
PREFIX const char * Spirit_Pinning() SUFFIX;

PREFIX const char * Spirit_Cuda() SUFFIX;
PREFIX const char * Spirit_OpenMP() SUFFIX;
PREFIX int Spirit_OpenMP_Get_Num_Threads() SUFFIX;
PREFIX const char * Spirit_Threads() SUFFIX;
PREFIX const char * Spirit_FFTW() SUFFIX;

#include "DLL_Undefine_Export.h"
#endif