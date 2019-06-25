

I/O
====================================================================

```C
#include "Spirit/IO.h"
```

TODO: give bool returns for these functions to indicate success?



Definition of file formats for vectorfields
--------------------------------------------------------------------

Spirit uses the OOMMF vector field file format with some minor variations.



### IO_Fileformat_OVF_bin

```C
IO_Fileformat_OVF_bin   0
```

OVF binary format, using the precision of Spirit



### IO_Fileformat_OVF_bin4

```C
IO_Fileformat_OVF_bin4  1
```

OVF binary format, using single precision



### IO_Fileformat_OVF_bin8

```C
IO_Fileformat_OVF_bin8  2
```

OVF binary format, using double precision



### IO_Fileformat_OVF_text

```C
IO_Fileformat_OVF_text  3
```

OVF text format



### IO_Fileformat_OVF_csv

```C
IO_Fileformat_OVF_csv   4
```

OVF text format with comma-separated columns



Other
--------------------------------------------------------------------



### IO_System_From_Config

```C
int IO_System_From_Config( State * state, const char * file, int idx_image=-1, int idx_chain=-1 )
```

Initialise a spin system using a config file.



### IO_Positions_Write

```C
void IO_Positions_Write( State * state, const char *file, int format=IO_Fileformat_OVF_bin, const char *comment = "-", int idx_image=-1, int idx_chain=-1 )
```

Write the spin positions as a vector field to file.



Spin configurations
--------------------------------------------------------------------



### IO_N_Images_In_File

```C
int IO_N_Images_In_File( State * state, const char *file, int idx_image=-1, int idx_chain=-1 )
```

Returns the number of images (i.e. OVF segments) in a given file.



### IO_Image_Read

```C
void IO_Image_Read( State *state, const char *file, int idx_image_infile=0, int idx_image_inchain=-1, int idx_chain=-1 )
```

Reads a spin configuration from a file.



### IO_Image_Write

```C
void IO_Image_Write( State *state, const char *file, int format=IO_Fileformat_OVF_bin, const char *comment = "-", int idx_image=-1, int idx_chain=-1 )
```

Writes a spin configuration to file.



### IO_Image_Append

```C
void IO_Image_Append( State *state, const char *file, int format=IO_Fileformat_OVF_bin, const char *comment = "-", int idx_image=-1, int idx_chain=-1 )
```

Appends a spin configuration to a file.



Chains
--------------------------------------------------------------------



### IO_Chain_Read

```C
void IO_Chain_Read( State *state, const char *file, int start_image_infile=0, int end_image_infile=-1, int insert_idx=0, int idx_chain=-1 )
```

Read a chain of spin configurations from a file.

If the current chain is not long enough to fit the file contents, systems
will be appended accordingly.



### IO_Chain_Write

```C
void IO_Chain_Write( State *state, const char *file, int format=IO_Fileformat_OVF_text, const char* comment = "-", int idx_chain=-1 )
```

Write the current chain of spin configurations to file



### IO_Chain_Append

```C
void IO_Chain_Append( State *state, const char *file, int format=IO_Fileformat_OVF_text, const char* comment = "-", int idx_chain=-1 )
```

Append the current chain of spin configurations to a file



Neighbours
--------------------------------------------------------------------



### IO_Image_Write_Neighbours_Exchange

```C
void IO_Image_Write_Neighbours_Exchange( State * state, const char * file, int idx_image=-1, int idx_chain=-1 )
```

Save the exchange interactions



### IO_Image_Write_Neighbours_DMI

```C
void IO_Image_Write_Neighbours_DMI( State * state, const char * file, int idx_image=-1, int idx_chain=-1 )
```

Save the DM interactions



Energies
--------------------------------------------------------------------



### IO_Image_Write_Energy_per_Spin

```C
void IO_Image_Write_Energy_per_Spin( State *state, const char *file, int format, int idx_image=-1, int idx_chain = -1 )
```

Save the spin-resolved energy contributions of a spin system



### IO_Image_Write_Energy

```C
void IO_Image_Write_Energy( State *state, const char *file, int idx_image=-1, int idx_chain=-1 )
```

Save the Energy contributions of a spin system



### IO_Chain_Write_Energies

```C
void IO_Chain_Write_Energies( State *state, const char *file, int idx_chain = -1 )
```

Save the Energy contributions of a chain of spin systems



### IO_Chain_Write_Energies_Interpolated

```C
void IO_Chain_Write_Energies_Interpolated( State *state, const char *file, int idx_chain = -1 )
```

Save the interpolated energies of a chain of spin systems



Eigenmodes
--------------------------------------------------------------------



### IO_Eigenmodes_Read

```C
void IO_Eigenmodes_Read( State *state, const char *file, int idx_image_inchain=-1, int idx_chain=-1 )
```

Read eigenmodes of a spin system from a file.

The file is expected to contain a chain of vector fields, which will
each be interpreted as one eigenmode of the system.



### IO_Eigenmodes_Write

```C
void IO_Eigenmodes_Write( State *state, const char *file, int format=IO_Fileformat_OVF_text, const char *comment = "-", int idx_image=-1, int idx_chain=-1 )
```

Write eigenmodes of a spin system to a file.

The file will contain a chain of vector fields corresponding to the
eigenmodes.
The eigenvalues of the respective modes will be written in a commment.

