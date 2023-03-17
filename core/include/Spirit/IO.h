#pragma once
#ifndef SPIRIT_CORE_IO_H
#define SPIRIT_CORE_IO_H
#include "DLL_Define_Export.h"

struct State;

/*
I/O
====================================================================

```C
#include "Spirit/IO.h"
```

TODO: give bool returns for these functions to indicate success?
*/

/*
Definition of file formats for vectorfields
--------------------------------------------------------------------

Spirit uses the OOMMF vector field file format with some minor variations.
*/

// OVF binary format, using the precision of Spirit
#define IO_Fileformat_OVF_bin 0

// OVF binary format, using single precision
#define IO_Fileformat_OVF_bin4 1

// OVF binary format, using double precision
#define IO_Fileformat_OVF_bin8 2

// OVF text format
#define IO_Fileformat_OVF_text 3

// OVF text format with comma-separated columns
#define IO_Fileformat_OVF_csv 4

/*
Other
--------------------------------------------------------------------
*/

// Initialise a spin system using a config file.
PREFIX int IO_System_From_Config( State * state, const char * file, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Write the spin positions as a vector field to file.
PREFIX void IO_Positions_Write(
    State * state, const char * file, int format = IO_Fileformat_OVF_bin, const char * comment = "-",
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Spin configurations
--------------------------------------------------------------------
*/

// Returns the number of images (i.e. OVF segments) in a given file.
PREFIX int IO_N_Images_In_File( State * state, const char * file, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Reads a spin configuration from a file.
PREFIX void IO_Image_Read(
    State * state, const char * file, int idx_image_infile = 0, int idx_image_inchain = -1, int idx_chain = -1 ) SUFFIX;

// Writes a spin configuration to file.
PREFIX void IO_Image_Write(
    State * state, const char * file, int format = IO_Fileformat_OVF_bin, const char * comment = "-",
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Appends a spin configuration to a file.
PREFIX void IO_Image_Append(
    State * state, const char * file, int format = IO_Fileformat_OVF_bin, const char * comment = "-",
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Chains
--------------------------------------------------------------------
*/

/*
Read a chain of spin configurations from a file.

If the current chain is not long enough to fit the file contents, systems
will be appended accordingly.
*/
PREFIX void IO_Chain_Read(
    State * state, const char * file, int start_image_infile = 0, int end_image_infile = -1, int insert_idx = 0,
    int idx_chain = -1 ) SUFFIX;

// Write the current chain of spin configurations to file
PREFIX void IO_Chain_Write(
    State * state, const char * file, int format = IO_Fileformat_OVF_text, const char * comment = "-",
    int idx_chain = -1 ) SUFFIX;

// Append the current chain of spin configurations to a file
PREFIX void IO_Chain_Append(
    State * state, const char * file, int format = IO_Fileformat_OVF_text, const char * comment = "-",
    int idx_chain = -1 ) SUFFIX;

/*
Neighbours
--------------------------------------------------------------------
*/

// Save the exchange interactions
PREFIX void
IO_Image_Write_Neighbours_Exchange( State * state, const char * file, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Save the DM interactions
PREFIX void
IO_Image_Write_Neighbours_DMI( State * state, const char * file, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Energies
--------------------------------------------------------------------
*/

// Save the spin-resolved energy contributions of a spin system
PREFIX void IO_Image_Write_Energy_per_Spin(
    State * state, const char * file, int format, int idx_image = -1, int idx_chain = -1 ) SUFFIX;
// Save the Energy contributions of a spin system
PREFIX void IO_Image_Write_Energy( State * state, const char * file, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

// Save the Energy contributions of a chain of spin systems
PREFIX void IO_Chain_Write_Energies( State * state, const char * file, int idx_chain = -1 ) SUFFIX;

// Save the interpolated energies of a chain of spin systems
PREFIX void IO_Chain_Write_Energies_Interpolated( State * state, const char * file, int idx_chain = -1 ) SUFFIX;

/*
Eigenmodes
--------------------------------------------------------------------
*/

/*
Read eigenmodes of a spin system from a file.

The file is expected to contain a chain of vector fields, which will
each be interpreted as one eigenmode of the system.
*/
PREFIX void
IO_Eigenmodes_Read( State * state, const char * file, int idx_image_inchain = -1, int idx_chain = -1 ) SUFFIX;

/*
Write eigenmodes of a spin system to a file.

The file will contain a chain of vector fields corresponding to the
eigenmodes.
The eigenvalues of the respective modes will be written in a commment.
*/
PREFIX void IO_Eigenmodes_Write(
    State * state, const char * file, int format = IO_Fileformat_OVF_text, const char * comment = "-",
    int idx_image = -1, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif