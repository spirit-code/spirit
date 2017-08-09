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
    
    int jmax;     // max iterations for Newton-Raphson loop
    int n;        // number of iteration after which the nCG will restart
    
    scalar tol_nCG, tol_NR;   // tolerances for optimizer and Newton-Raphson
    scalar eps_nCG, eps_NR;   // Newton-Raphson and optimizer tolerance squared
    
    bool restart_nCG, continue_NR;  // conditions for restarting nCG or continuing Newton-Raphson 
    
    // step sizes
    std::vector<scalarfield> alpha, beta;
    
    // XXX: right type might be std::vector<scalar> and NOT std::vector<scalarfield>
    // delta scalarfields
    std::vector<scalarfield> delta_0, delta_new, delta_old, delta_d;
    
    // residual and new configuration states
    std::vector<vectorfield> res, d;

    // buffer variables for checking convergence for optimizer and Newton-Raphson
    std::vector<scalarfield> r_dot_d, dda2;
    
  };
}
#endif