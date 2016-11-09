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

// From Config File
DLLEXPORT int IO_System_From_Config(State * state, const char * file, int idx_image=-1, int idx_chain=-1);

///// TODO: give bool returns for these functions to indicate success??

// Images
DLLEXPORT void IO_Image_Read(State * state, const char * file, int format=IO_Fileformat_Regular, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Image_Write(State * state, const char * file, int format=IO_Fileformat_Regular, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Image_Append(State * state, const char * file, int iteration=0, int format=IO_Fileformat_Regular, int idx_image=-1, int idx_chain=-1);

// Chains
DLLEXPORT void IO_Chain_Read(State * state, const char * file, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Chain_Write(State * state, const char * file, int idx_image=-1, int idx_chain=-1);

// Collection
DLLEXPORT void IO_Collection_Read(State * state, const char * file, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void IO_Collection_Write(State * state, const char * file, int idx_image=-1, int idx_chain=-1);

// Data
DLLEXPORT void IO_Energies_Save(State * state, const char * file, int idx_chain = -1);
DLLEXPORT void IO_Energies_Spins_Save(State * state, const char * file, int idx_chain = -1);
DLLEXPORT void IO_Energies_Interpolated_Save(State * state, const char * file, int idx_chain = -1);

#include "DLL_Undefine_Export.h"
#endif