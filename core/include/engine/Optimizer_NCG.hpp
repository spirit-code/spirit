#pragma once
#ifndef OPTIMIZER_NCG_H
#define OPTIMIZER_NCG_H

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Optimizer.hpp>

namespace Engine
{
  /*
    Nonlinear Conjugate Gradient Optimizer
      See "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain" by Jonathan 
      Richard Shewchuk p.51
  */
  
  class Optimizer_NCG : public Optimizer
  {
  public:
    Optimizer_NCG( std::shared_ptr<Engine::Method> method );
    
    // One Iteration
    void Iteration() override;
    
    // Optimizer name as string
    std::string Name() override;
    std::string FullName() override;
  
  private:
    // check if the Newton-Raphson has converged
    bool NR_converged();
    
    // max iterations for Newton-Raphson loop
    int jmax;
    
    // tolerances for optimizer and Newton-Raphson
    scalar tol_nGB, tol_NR;
    
    // step sizes
    scalar alpha, beta;
    
    scalarfield delta_0, delta_new, delta_old, delta_d;
    
    // residual and new configuration state
    std::vector<vectorfield> res, d, x;
    
  };
}
#endif