#pragma once
#ifndef INTERFACE_CHAIN_H
#define INTERFACE_CHAIN_H

#include "Interface_State.h"

// Move between images (change active_image)
extern "C" void Chain_next_Image(State * state);
extern "C" void Chain_prev_Image(State * state);

// Insert/Replace/Delete images
extern "C" void Chain_Insert_Image_Before(State * state, Data::Spin_System & image);
extern "C" void Chain_Insert_Image_After(State * state, Data::Spin_System & image);
extern "C" void Chain_Replace_Image(State * state, Data::Spin_System & image);
extern "C" void Chain_Delete_Image(State * state, int idx);

// Update Data (primarily for plots)
extern "C" void Chain_Update_Data(State * state);
extern "C" void Chain_Setup_Data(State * state);

// TODO: how to handle shared pointers and vectors??

// extern "C" void Chain_New_Images(State * state, int idx_chain, std::vector<std::shared_ptr<Data::Spin_System>> images);

// extern "C" void Chain_from_File(State * state, int idx_chain, const char * filename);

// extern "C" void Chain_Insert_Image_Right(State * state, int idx_chain, int idx_image, std::shared_ptr<Data::Spin_System> image)

#endif