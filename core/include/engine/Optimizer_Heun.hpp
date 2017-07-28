#pragma once
#ifndef OPTIMIZER_HEUN_H
#define OPTIMIZER_HEUN_H

#include <vector>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Optimizer.hpp>
#include <utility/Constants.hpp>
#include <engine/Method.hpp>

namespace Engine
{
  /*
  The Heun method is a direct optimization of a Spin System:
  The Spin System will follow the applied force on a direct trajectory, which need not be physical.
  See also https://en.wikipedia.org/wiki/Heun%27s_method
  */
  class Optimizer_Heun : public Optimizer
  {
  public:
    Optimizer_Heun(std::shared_ptr<Engine::Method> method);
    
    // One Iteration
    void Iteration() override;
    
    // Optimizer name as string
    std::string Name() override;
    std::string FullName() override;
    
  private:
    
    // Temporary Spins arrays
    std::vector<std::shared_ptr<vectorfield>> spins_temp, spins_predictor;
    vectorfield temp1, temp2;
    
    // Virtual Heun Forces used in the Steps
    std::vector<vectorfield> virtualforce;
    
    // pointer to spin system
    std::shared_ptr<Data::Spin_System> s;
    
    // method time step
    scalar dt;
    
    void VirtualForce( const vectorfield & spins, 
                       const Data::Parameters_Method_LLG & llg_params, 
                       const vectorfield & effective_field,  
                       vectorfield & force );
    
  };
}

#endif