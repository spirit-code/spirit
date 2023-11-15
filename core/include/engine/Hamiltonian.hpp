#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_HPP

#include <engine/interaction/Hamiltonian_Base.hpp>
#include <engine/interaction/Hamiltonian_Gaussian.hpp>
#include <engine/interaction/Hamiltonian_Heisenberg.hpp>
#ifdef SPIRIT_USE_CUDA
#include <engine/interaction/Hamiltonian_Heisenberg_Kernels.cuh>
#endif

#endif
