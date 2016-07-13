#pragma once
#ifndef INTERFACE_GLOBALS_H
#define INTERFACE_GLOBALS_H

#include "Spin_System_Chain.h"
#include "Optimizer.h"


struct State {
  std::shared_ptr<Data::Spin_System_Chain> c;
  std::shared_ptr<Engine::Optimizer> optim;
  int active_image;
};

#endif