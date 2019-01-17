

Chain
====================================================================

```C
#include "Spirit/Chain.h"
```

A chain of spin systems can be used for example for
- calculating minimum energy paths using the GNEB method
- running multiple (e.g. LLG) calculations in parallel
- calculate HTST transition rates



### Chain_Get_NOI

```C
int Chain_Get_NOI(State * state, int idx_chain=-1)
```

Returns the number of images in the chain.



Change the active image
--------------------------------------------------------------------



### Chain_next_Image

```C
bool Chain_next_Image(State * state, int idx_chain=-1)
```

Move to next image in the chain (change active_image).

**Returns:** success of the function



### Chain_prev_Image

```C
bool Chain_prev_Image(State * state, int idx_chain=-1)
```

Move to previous image in the chain (change active_image)

**Returns:** success of the function



### Chain_Jump_To_Image

```C
bool Chain_Jump_To_Image(State * state, int idx_image=-1, int idx_chain=-1)
```

Move to a specific image (change active_image)

**Returns:** success of the function



Insert/replace/delete images
--------------------------------------------------------------------



### Chain_Set_Length

```C
void Chain_Set_Length(State * state, int n_images, int idx_chain=-1)
```

Set the number of images in the chain.

If it is currently less, a corresponding number of images will be appended.
If it is currently more, a corresponding number of images will be deleted from the end.

**Note:** you need to have an image in the *clipboard*.



### Chain_Image_to_Clipboard

```C
void Chain_Image_to_Clipboard(State * state, int idx_image=-1, int idx_chain=-1)
```

Copy an image from the chain (default: active image).

You can later insert it anywhere in the chain.



### Chain_Replace_Image

```C
void Chain_Replace_Image(State * state, int idx_image=-1, int idx_chain=-1)
```

Replaces the specified image (default: active image).

**Note:** you need to have an image in the *clipboard*.



### Chain_Insert_Image_Before

```C
void Chain_Insert_Image_Before(State * state, int idx_image=-1, int idx_chain=-1)
```

Inserts an image in front of the specified image index (default: active image).

**Note:** you need to have an image in the *clipboard*.



### Chain_Insert_Image_After

```C
void Chain_Insert_Image_After(State * state, int idx_image=-1, int idx_chain=-1)
```

Inserts an image behind the specified image index (default: active image).

**Note:** you need to have an image in the *clipboard*.



### Chain_Push_Back

```C
void Chain_Push_Back(State * state, int idx_chain=-1)
```

Appends an image to the chain.

**Note:** you need to have an image in the *clipboard*.



### Chain_Delete_Image

```C
bool Chain_Delete_Image(State * state, int idx_image=-1, int idx_chain=-1)
```

Removes an image from the chain (default: active image).

Does nothing if the chain contains only one image.



### Chain_Pop_Back

```C
bool Chain_Pop_Back(State * state, int idx_chain=-1)
```

Removes the last image of the chain.

Does nothing if the chain contains only one image.

**Returns:** success of the operation



Calculate data
--------------------------------------------------------------------



### Chain_Get_Rx

```C
void Chain_Get_Rx(State * state, float * Rx, int idx_chain=-1)
```

Fills an array with the reaction coordinate values of the images in the chain.



### Chain_Get_Rx_Interpolated

```C
void Chain_Get_Rx_Interpolated(State * state, float * Rx_interpolated, int idx_chain=-1)
```

Fills an array with the interpolated reaction coordinate values along the chain.



### Chain_Get_Energy

```C
void Chain_Get_Energy(State * state, float * energy, int idx_chain=-1)
```

Fills an array with the energy values of the images in the chain.



### Chain_Get_Energy_Interpolated

```C
void Chain_Get_Energy_Interpolated(State * state, float * E_interpolated, int idx_chain=-1)
```

Fills an array with the interpolated energy values along the chain.



TODO: energy array getter
std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated(State * state, int idx_chain=-1) SUFFIX;



### Chain_Get_HTST_Info

```C
PREFIX void Chain_Get_HTST_Info( State * state, float * eigenvalues_min, float * eigenvalues_sp, float * temperature_exponent, float * me, float * Omega_0, float * s, float * volume_min, float * volume_sp, float * prefactor_dynamical, float * prefactor, int idx_chain=-1 ) SUFFIX
```

Retrieves a set of arrays and single values from HTST:
- eigenvalues_min: eigenvalues at the minimum (array of length 2*NOS)
- eigenvalues_sp: eigenvalues at the saddle point (array of length 2*NOS)
- temperature_exponent: the exponent of the temperature-dependent prefactor
- me:
- Omega_0:
- s:
- volume_min: zero mode volume at the minimum
- volume_sp: zero mode volume at the saddle point
- prefactor_dynamical: the dynamical rate prefactor
- prefactor: the total rate prefactor for the transition

**Note:** for this function to retrieve anything, you need to first calculate the data



### Chain_Update_Data

```C
void Chain_Update_Data(State * state, int idx_chain=-1)
```

Update Data, such as energy or reaction coordinate.

This is primarily used for the plotting in the GUI, but needs to be
called e.g. before calling the automatic setting of GNEB image types.



### Chain_Setup_Data

```C
void Chain_Setup_Data(State * state, int idx_chain=-1)
```

You probably won't need this.

