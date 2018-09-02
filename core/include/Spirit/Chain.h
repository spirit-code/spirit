#pragma once
#ifndef INTERFACE_CHAIN_H
#define INTERFACE_CHAIN_H
#include "DLL_Define_Export.h"

struct State;

// Info
DLLEXPORT int Chain_Get_NOI(State * state, int idx_chain=-1) noexcept;

// Move between images (change active_image)
DLLEXPORT bool Chain_next_Image(State * state, int idx_chain=-1) noexcept;
DLLEXPORT bool Chain_prev_Image(State * state, int idx_chain=-1) noexcept;
DLLEXPORT bool Chain_Jump_To_Image(State * state, int idx_image=-1, int idx_chain=-1) noexcept;

// Insert/Replace/Delete images
DLLEXPORT void Chain_Set_Length(State * state, int n_images, int idx_chain=-1) noexcept;
DLLEXPORT void Chain_Image_to_Clipboard(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Chain_Replace_Image(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Chain_Insert_Image_Before(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Chain_Insert_Image_After(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT void Chain_Push_Back(State * state, int idx_chain=-1) noexcept;
DLLEXPORT bool Chain_Delete_Image(State * state, int idx_image=-1, int idx_chain=-1) noexcept;
DLLEXPORT bool Chain_Pop_Back(State * state, int idx_chain=-1) noexcept;

// Get Data
DLLEXPORT void Chain_Get_Rx(State * state, float * Rx, int idx_chain = -1) noexcept;
DLLEXPORT void Chain_Get_Rx_Interpolated(State * state, float * Rx_interpolated, int idx_chain = -1) noexcept;
DLLEXPORT void Chain_Get_Energy(State * state, float * energy, int idx_chain = -1) noexcept;
DLLEXPORT void Chain_Get_Energy_Interpolated(State * state, float * E_interpolated, int idx_chain = -1) noexcept;
// TODO: energy array getter
// std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated(State * state, int idx_chain=-1) noexcept;

// Update Data (primarily for plots)
DLLEXPORT void Chain_Update_Data(State * state, int idx_chain=-1) noexcept;
DLLEXPORT void Chain_Setup_Data(State * state, int idx_chain=-1) noexcept;

#include "DLL_Undefine_Export.h"
#endif