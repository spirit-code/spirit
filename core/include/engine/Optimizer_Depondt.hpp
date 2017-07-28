#pragma once
#ifndef OPTIMIZER_DEPONDT_H
#define OPTIMIZER_DEPONDT_H

#include <vector>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Optimizer.hpp>
#include <utility/Constants.hpp>
#include <engine/Method.hpp>

namespace Engine
{
  class Optimizer_Depondt : public Optimizer
  {
  public:
    Optimizer_Depondt( std::shared_ptr<Engine::Method> method );
    
    void Iteration() override;
    
    std::string Name() override;
    std::string FullName() override;
    
  private:
    scalar dtg;
    
    // Temporary spins arrays
    std::vector<std::shared_ptr<vectorfield>> spins_temp;
    std::vector<std::shared_ptr<vectorfield>> spins_predictor;
    
    // Virtual force
    std::vector<vectorfield> virtualforce;
    std::vector<vectorfield> virtualforce_predictor;
    std::vector<vectorfield> rotationaxis;

    std::vector<scalarfield> virtualforce_norm;
    
    // preccession angle
    scalarfield angle;
    
    std::shared_ptr<Data::Spin_System> s;
    
    void VirtualForce( const vectorfield & spins, 
                       const Data::Parameters_Method_LLG & llg_params, 
                       const vectorfield & effective_field,  
                       vectorfield & force );
  };
}

#endif