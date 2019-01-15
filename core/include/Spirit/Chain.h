#pragma once
#ifndef INTERFACE_CHAIN_H
#define INTERFACE_CHAIN_H
#include "DLL_Define_Export.h"

struct State;

// Info
PREFIX int Chain_Get_NOI(State * state, int idx_chain=-1) SUFFIX;

// Move between images (change active_image)
PREFIX bool Chain_next_Image(State * state, int idx_chain=-1) SUFFIX;
PREFIX bool Chain_prev_Image(State * state, int idx_chain=-1) SUFFIX;
PREFIX bool Chain_Jump_To_Image(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;

// Insert/Replace/Delete images
PREFIX void Chain_Set_Length(State * state, int n_images, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Image_to_Clipboard(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Replace_Image(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Insert_Image_Before(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Insert_Image_After(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Push_Back(State * state, int idx_chain=-1) SUFFIX;
PREFIX bool Chain_Delete_Image(State * state, int idx_image=-1, int idx_chain=-1) SUFFIX;
PREFIX bool Chain_Pop_Back(State * state, int idx_chain=-1) SUFFIX;

// Get Data
PREFIX void Chain_Get_Rx(State * state, float * Rx, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Get_Rx_Interpolated(State * state, float * Rx_interpolated, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Get_Energy(State * state, float * energy, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Get_Energy_Interpolated(State * state, float * E_interpolated, int idx_chain=-1) SUFFIX;
// TODO: energy array getter
// std::vector<std::vector<float>> Chain_Get_Energy_Array_Interpolated(State * state, int idx_chain=-1) SUFFIX;

PREFIX void Chain_Get_HTST_Info( State * state, float * eigenvalues_min, float * eigenvalues_sp, float * temperature_exponent,
                          float * me, float * Omega_0, float * s, float * volume_min, float * volume_sp,
                          float * prefactor_dynamical, float * prefactor, int idx_chain=-1 ) SUFFIX;

// Update Data (primarily for plots)
PREFIX void Chain_Update_Data(State * state, int idx_chain=-1) SUFFIX;
PREFIX void Chain_Setup_Data(State * state, int idx_chain=-1) SUFFIX;

#include "DLL_Undefine_Export.h"
#endif