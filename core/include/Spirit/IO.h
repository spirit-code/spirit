#pragma once
#ifndef INTERFACE_IO_H
#define INTERFACE_IO_H
#include "DLL_Define_Export.h"
struct State;

// Define File Formats for Vector Fields
#define IO_Fileformat_OVF_bin   0
#define IO_Fileformat_OVF_bin4  1
#define IO_Fileformat_OVF_bin8  2
#define IO_Fileformat_OVF_text  3
#define IO_Fileformat_OVF_csv   4

// From Config File
DLLEXPORT int IO_System_From_Config( State * state, const char * file, int idx_image=-1,
                                     int idx_chain=-1 ) noexcept;

// Geometry
DLLEXPORT void IO_Positions_Write( State * state, const char *file, int format=IO_Fileformat_OVF_bin, 
                                   const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) noexcept;

///// TODO: give bool returns for these functions to indicate success??

// Images
DLLEXPORT int IO_N_Images_In_File( State * state, const char *file, int idx_image=-1, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Read( State *state, const char *file, int idx_image_infile=0, 
                              int idx_image_inchain=-1, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Write( State *state, const char *file, int format=IO_Fileformat_OVF_bin, 
                               const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Append( State *state, const char *file, int format=IO_Fileformat_OVF_bin, 
                                const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) noexcept;

// Chains
DLLEXPORT void IO_Chain_Read( State *state, const char *file, int start_image_infile=0, 
                              int end_image_infile=-1, int insert_idx=0, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Chain_Write( State *state, const char *file, int format, 
                               const char* comment = "-", int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Chain_Append( State *state, const char *file, int format, 
                                const char* comment = "-", int idx_chain=-1 ) noexcept;

// Save the interactions
DLLEXPORT void IO_Image_Write_Neighbours_Exchange( State * state, const char * file, 
                                         int idx_image=-1, int idx_chain=-1 ) noexcept;
DLLEXPORT void IO_Image_Write_Neighbours_DMI( State * state, const char * file, 
                                    int idx_image=-1, int idx_chain=-1 ) noexcept;

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

#include "DLL_Undefine_Export.h"
#endif
