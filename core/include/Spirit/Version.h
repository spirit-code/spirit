#pragma once
#ifndef VERSION_H
#define VERSION_H
#include "DLL_Define_Export.h"

PREFIX const int Spirit_Version_Major() SUFFIX;
PREFIX const int Spirit_Version_Minor() SUFFIX;
PREFIX const int Spirit_Version_Patch() SUFFIX;

PREFIX const char * Spirit_Version() SUFFIX;
PREFIX const char * Spirit_Version_Revision() SUFFIX;
PREFIX const char * Spirit_Version_Full() SUFFIX;

#include "DLL_Undefine_Export.h"
#endif