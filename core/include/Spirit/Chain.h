#pragma once
#ifndef SPIRIT_CORE_CHAIN_H
#define SPIRIT_CORE_CHAIN_H
#include "DLL_Define_Export.h"

struct State;

/*
Chain
====================================================================

```C
#include "Spirit/Chain.h"
```

A chain of spin systems can be used for example for
- calculating minimum energy paths using the GNEB method
- running multiple (e.g. LLG) calculations in parallel
*/

// Returns the number of images in the chain.
PREFIX int Chain_Get_NOI( State * state, int idx_chain = -1 ) SUFFIX;

/*
Change the active image
--------------------------------------------------------------------
*/

/*
Move to next image in the chain (change active_image).

**Returns:** success of the function
*/
PREFIX bool Chain_next_Image( State * state, int idx_chain = -1 ) SUFFIX;

/*
Move to previous image in the chain (change active_image)

**Returns:** success of the function
*/
PREFIX bool Chain_prev_Image( State * state, int idx_chain = -1 ) SUFFIX;

/*
Move to a specific image (change active_image)

**Returns:** success of the function
*/
PREFIX bool Chain_Jump_To_Image( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Insert/replace/delete images
--------------------------------------------------------------------
*/

/*
Set the number of images in the chain.

If it is currently less, a corresponding number of images will be appended.
If it is currently more, a corresponding number of images will be deleted from the end.

**Note:** you need to have an image in the *clipboard*.
*/
PREFIX void Chain_Set_Length( State * state, int n_images, int idx_chain = -1 ) SUFFIX;

/*
Copy an image from the chain (default: active image).

You can later insert it anywhere in the chain.
*/
PREFIX void Chain_Image_to_Clipboard( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Replaces the specified image (default: active image).

**Note:** you need to have an image in the *clipboard*.
*/
PREFIX void Chain_Replace_Image( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Inserts an image in front of the specified image index (default: active image).

**Note:** you need to have an image in the *clipboard*.
*/
PREFIX void Chain_Insert_Image_Before( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Inserts an image behind the specified image index (default: active image).

**Note:** you need to have an image in the *clipboard*.
*/
PREFIX void Chain_Insert_Image_After( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Appends an image to the chain.

**Note:** you need to have an image in the *clipboard*.
*/
PREFIX void Chain_Push_Back( State * state, int idx_chain = -1 ) SUFFIX;

/*
Removes an image from the chain (default: active image).

Does nothing if the chain contains only one image.
*/
PREFIX bool Chain_Delete_Image( State * state, int idx_image = -1, int idx_chain = -1 ) SUFFIX;

/*
Removes the last image of the chain.

Does nothing if the chain contains only one image.

**Returns:** success of the operation
*/
PREFIX bool Chain_Pop_Back( State * state, int idx_chain = -1 ) SUFFIX;

/*
Calculate data
--------------------------------------------------------------------
*/

/*
Fills an array with the reaction coordinate values of the images in the chain.
*/
PREFIX void Chain_Get_Rx( State * state, float * Rx, int idx_chain = -1 ) SUFFIX;

/*
Fills an array with the interpolated reaction coordinate values along the chain.
*/
PREFIX void Chain_Get_Rx_Interpolated( State * state, float * Rx_interpolated, int idx_chain = -1 ) SUFFIX;

/*
Fills an array with the energy values of the images in the chain.
*/
PREFIX void Chain_Get_Energy( State * state, float * energy, int idx_chain = -1 ) SUFFIX;

/*
Fills an array with the interpolated energy values along the chain.
*/
PREFIX void Chain_Get_Energy_Interpolated( State * state, float * E_interpolated, int idx_chain = -1 ) SUFFIX;

/* TODO: energy array getter
std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated(State * state, int idx_chain=-1) SUFFIX;
*/

/*
Update Data, such as energy or reaction coordinate.

This is primarily used for the plotting in the GUI, but needs to be
called e.g. before calling the automatic setting of GNEB image types.
*/
PREFIX void Chain_Update_Data( State * state, int idx_chain = -1 ) SUFFIX;

/*
You probably won't need this.
*/
PREFIX void Chain_Setup_Data( State * state, int idx_chain = -1 ) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif