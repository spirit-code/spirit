#pragma once
#ifndef INTERFACE_GLOBALS_H
#define INTERFACE_GLOBALS_H

#include "Spin_System_Chain.h"
#include "Optimizer.h"

extern std::shared_ptr<Data::Spin_System_Chain> c;
extern std::shared_ptr<Engine::Optimizer> optim;
extern int active_image;

#endif