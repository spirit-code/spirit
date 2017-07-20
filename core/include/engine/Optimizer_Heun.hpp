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
    std::vector<std::shared_ptr<vectorfield>> spins_temp;
    vectorfield temp1, temp2, spins_predictor;
    
    // Virtual Heun Forces used in the Steps
    std::vector<vectorfield> virtualforce;
    
    // pointer to spin system
    std::shared_ptr<Data::Spin_System> s;
    
    // method time step
    scalar dt;
    
    // TODO: for dynamic solver we would need a damping factor alpha
    
    // TODO: THE HEUN METHOD CAN BE REWRITTEN TO BE NICER:
    
    // Calculate the virtual Heun force to be used in the Steps
    // old comment function 
    /*
    void VirtualForce( const int nos, 
                       const std::vector<scalar> & spins, 
                       const std::vector<scalar> & beff, 
                       const scalar dt, 
                       std::vector<scalar> & force ) ;
    */
    void VirtualForce( const vectorfield & spins, 
                       const Data::Parameters_Method_LLG & llg_params, 
                       const vectorfield & effective_field,  
                       vectorfield & force );

    // First Part of one Optimization step
    // void FirstStep( const int nos, std::vector<scalar> & spins, scalar dt, 
    // std::vector<scalar> & force, std::vector<scalar> & spins_temp );
    
    // Second Part of one Optimization step
    // void SecondStep( const int nos, std::vector<scalar> & spins, scalar dt, 
    // std::vector<scalar> & force );
    
  };
}

#endif