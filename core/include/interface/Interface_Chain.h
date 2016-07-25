#pragma once
#ifndef INTERFACE_CONFIGURATIONS_H
#define INTERFACE_CONFIGURATIONS_H

#include "Interface_State.h"

// Move between images (change active_image)
extern "C" void Chain_next_Image(State * state);
extern "C" void Chain_prev_Image(State * state);

// TODO: Place these into Interface_Collection (Spin_System_Chain_Collection)
// extern "C" void Collection_next_Chain(State * state);
// extern "C" void Collection_prev_Chain(State * state);


// TODO: how to handle shared pointers and vectors??

// extern "C" void Chain_New_Images(State * state, int idx_chain, std::vector<std::shared_ptr<Data::Spin_System>> images);

// extern "C" void Chain_from_File(State * state, int idx_chain, const char * filename);

// extern "C" void Chain_Insert_Image_Right(State * state, int idx_chain, int idx_image, std::shared_ptr<Data::Spin_System> image)

#endif