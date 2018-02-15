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
#define IO_Fileformat_OVF_bin8      4   // 
#define IO_Fileformat_OVF_bin4      5   // OOMF Vector Field (OVF2.0) file format
#define IO_Fileformat_OVF_text      6   // 

// From Config File
DLLEXPORT int IO_System_From_Config( State * state, const char * file, int idx_image=-1,
                                     int idx_chain=-1 ) noexcept;

// Geometry
DLLEXPORT void IO_Positions_Write( State * state, const char *file, int format=IO_Fileformat_Regular, 
                                   const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) noexcept;

///// TODO: give bool returns for these functions to indicate success??

// Images
DLLEXPORT int IO_N_Images_In_File( State *state, const char *file, 
                                   int format=IO_Fileformat_Regular, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Read( State *state, const char *file, int format=IO_Fileformat_Regular, 
                              int idx_image_infile=-1, int idx_image_inchain=-1, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Write( State *state, const char *file, int format=IO_Fileformat_Regular, 
                               const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Append( State *state, const char *file, int format=IO_Fileformat_Regular, 
                                const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) noexcept;

// Chains
DLLEXPORT void IO_Chain_Read( State *state, const char *file, int format=IO_Fileformat_Regular, 
                              int starting_image=-1, int ending_image=-1, int insert_idx=-1, 
                              int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Chain_Write( State *state, const char *file, int format, 
                               const char* comment = "-", int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Chain_Append( State *state, const char *file, int format, 
                                const char* comment = "-", int idx_chain=-1 ) noexcept;

// Save the spin-resolved energy contributions of a spin system
DLLEXPORT void IO_Image_Write_Energy_per_Spin( State *state, const char *file, int idx_image=-1, 
                                               int idx_chain = -1 ) noexcept;
// Save the Energy contributions of a spin system
DLLEXPORT void IO_Image_Write_Energy( State *state, const char *file, int idx_image=-1, 
                                      int idx_chain=-1 ) noexcept;

// Save the Energy contributions of a chain of spin systems
DLLEXPORT void IO_Chain_Write_Energies( State *state, const char *file, int idx_chain = -1 ) noexcept;
// Save the interpolated energies of a chain of spin systems
DLLEXPORT void IO_Chain_Write_Energies_Interpolated( State *state, const char *file, 
                                                     int idx_chain = -1 ) noexcept;

// Collection
DLLEXPORT void IO_Collection_Read( State *state, const char *file, int idx_image=-1, 
                                   int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Collection_Write( State *state, const char *file, int idx_image=-1, 
                                    int idx_chain=-1 ) noexcept;

#include "DLL_Undefine_Export.h"
#endif