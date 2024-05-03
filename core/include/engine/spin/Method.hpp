#pragma once

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Method.hpp>
#include <engine/spin/Hamiltonian.hpp>

namespace Engine
{

namespace Spin
{

using hamiltonian_t = Engine::Spin::HamiltonianVariant;
using system_t = Data::Spin_System<hamiltonian_t>;
using chain_t  = Data::Spin_System_Chain<hamiltonian_t>;

}

}
