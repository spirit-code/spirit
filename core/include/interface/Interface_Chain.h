#pragma once
#ifndef INTERFACE_CHAIN_H
#define INTERFACE_CHAIN_H
#include "DLL_Define_Export.h"

// TODO: do without vector!
#include <vector>

struct State;

// Info
DLLEXPORT int Chain_Get_Index(State * state);
DLLEXPORT int Chain_Get_NOI(State * state, int idx_chain=-1);

// Move between images (change active_image)
DLLEXPORT bool Chain_next_Image(State * state, int idx_chain=-1);
DLLEXPORT bool Chain_prev_Image(State * state, int idx_chain=-1);

// Insert/Replace/Delete images
DLLEXPORT void Chain_Image_to_Clipboard(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Chain_Insert_Image_Before(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Chain_Insert_Image_After(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT void Chain_Replace_Image(State * state, int idx_image=-1, int idx_chain=-1);
DLLEXPORT bool Chain_Delete_Image(State * state, int idx_image=-1, int idx_chain=-1);

// Get Data
std::vector<float> Chain_Get_Rx(State * state, int idx_chain=-1);
std::vector<float> Chain_Get_Rx_Interpolated(State * state, int idx_chain = -1);
std::vector<float> Chain_Get_Energy_Interpolated(State * state, int idx_chain = -1);
std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated(State * state, int idx_chain=-1);

// Update Data (primarily for plots)
DLLEXPORT void Chain_Update_Data(State * state, int idx_chain=-1);
DLLEXPORT void Chain_Setup_Data(State * state, int idx_chain=-1);

// TODO: file read
//DLLEXPORT void Chain_from_File(State * state, const char * filename, int idx_chain=-1);

#include "DLL_Undefine_Export.h"
#endif