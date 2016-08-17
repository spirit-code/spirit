#pragma once
#ifndef INTERFACE_CHAIN_H
#define INTERFACE_CHAIN_H
struct State;

// Info
extern "C" int Chain_Get_Index(State * state);
extern "C" int Chain_Get_NOI(State * state, int idx_chain=-1);

// Move between images (change active_image)
extern "C" void Chain_next_Image(State * state, int idx_chain=-1);
extern "C" void Chain_prev_Image(State * state, int idx_chain=-1);

// Insert/Replace/Delete images
extern "C" void Chain_Image_to_Clipboard(State * state, int idx_image=-1, int idx_chain=-1);
extern "C" void Chain_Insert_Image_Before(State * state, int idx_image=-1, int idx_chain=-1);
extern "C" void Chain_Insert_Image_After(State * state, int idx_image=-1, int idx_chain=-1);
extern "C" void Chain_Replace_Image(State * state, int idx_image=-1, int idx_chain=-1);
extern "C" void Chain_Delete_Image(State * state, int idx_image=-1, int idx_chain=-1);

// Update Data (primarily for plots)
extern "C" void Chain_Update_Data(State * state, int idx_chain=-1);
extern "C" void Chain_Setup_Data(State * state, int idx_chain=-1);

// TODO: Get reaction coordinates and interpolated energies

// TODO: file read
//extern "C" void Chain_from_File(State * state, const char * filename, int idx_chain=-1);

#endif