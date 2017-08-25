#pragma once
#ifndef INTERFACE_IO_H
#define INTERFACE_IO_H
#include "DLL_Define_Export.h"
struct State;

// Define File Formats for Vector Fields
#define IO_Fileformat_Regular       0   // sx sy sz (separated by whitespace)
#define IO_Fileformat_Regular_Pos   1   // px py pz sx sy sz (separated by whitespace)
#define IO_Fileformat_CSV           2   // sx, sy, sz (separated by commas)
#define IO_Fileformat_CSV_Pos       3   // px, py, pz, sx, sy, (sz separated by commas)
#define IO_Fileformat_OVF           4   // OOMF Vector Field (OVF) file format

// From Config File
DLLEXPORT int IO_System_From_Config(State * state, const char * file, int idx_image=-1, int idx_chain=-1);

///// TODO: give bool returns for these functions to indicate success??

// Images
DLLEXPORT void IO_Image_Read(State * state, const char * file, int format=IO_Fileformat_Regular, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Image_Write(State * state, const char * file, int format=IO_Fileformat_Regular, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Image_Append(State * state, const char * file, int iteration=0, int format=IO_Fileformat_Regular, int idx_image=-1, int idx_chain=-1);

// Chains
DLLEXPORT void IO_Chain_Read(State * state, const char * file, int format=IO_Fileformat_Regular, int idx_chain=-1);
DLLEXPORT void IO_Chain_Write(State * state, const char * file, int format=IO_Fileformat_Regular, int idx_chain=-1);

// Collection
DLLEXPORT void IO_Collection_Read(State * state, const char * file, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Collection_Write(State * state, const char * file, int idx_image=-1, int idx_chain=-1);


// Save the spin-resolved energy contributions of a spin system
DLLEXPORT void IO_Write_System_Energy_per_Spin(State * state, const char * file, int idx_chain = -1);
// Save the Energy contributions of a spin system
DLLEXPORT void IO_Write_System_Energy(State * state, const char * file, int idx_image=-1, int idx_chain=-1);

// Save the Energy contributions of a chain of spin systems
DLLEXPORT void IO_Write_Chain_Energies(State * state, const char * file, int idx_chain = -1);
// Save the interpolated energies of a chain of spin systems
DLLEXPORT void IO_Write_Chain_Energies_Interpolated(State * state, const char * file, int idx_chain = -1);


#include "DLL_Undefine_Export.h"
#endif