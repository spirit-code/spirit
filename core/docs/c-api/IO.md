

I/O
====================================================================

TODO: give bool returns for these functions to indicate success?



Define File Formats for Vector Fields
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

From Config File



### IO_Positions_Write

```C
PREFIX void IO_Positions_Write( State * state, const char *file, int format=IO_Fileformat_OVF_bin, const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) SUFFIX
```

Geometry



Spin configurations
--------------------------------------------------------------------



### IO_N_Images_In_File

```C
int IO_N_Images_In_File( State * state, const char *file, int idx_image=-1, int idx_chain=-1 )
```

Images



Chains
--------------------------------------------------------------------



### IO_Chain_Read

```C
PREFIX void IO_Chain_Read( State *state, const char *file, int start_image_infile=0, int end_image_infile=-1, int insert_idx=0, int idx_chain=-1 ) SUFFIX
```

Chains



Neighbours
--------------------------------------------------------------------



### IO_Image_Write_Neighbours_Exchange

```C
void IO_Image_Write_Neighbours_Exchange( State * state, const char * file, int idx_image=-1, int idx_chain=-1 )
```

Save the interactions



Energies
--------------------------------------------------------------------



### IO_Image_Write_Energy_per_Spin

```C
void IO_Image_Write_Energy_per_Spin( State *state, const char *file, int idx_image=-1, int idx_chain = -1 )
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
PREFIX void IO_Eigenmodes_Write( State *state, const char *file, int format=IO_Fileformat_OVF_text, const char *comment = "-", int idx_image=-1, int idx_chain=-1 ) SUFFIX
```

Write eigenmodes of a spin system to a file.

The file will contain a chain of vector fields corresponding to the
eigenmodes.
The eigenvalues of the respective modes will be written in a commment.

