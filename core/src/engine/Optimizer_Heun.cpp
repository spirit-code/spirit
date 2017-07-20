#include <engine/Optimizer_Heun.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Dense>

using namespace Utility;

namespace Engine
{
  Optimizer_Heun::Optimizer_Heun(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
  {
    this->virtualforce = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->spins_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      spins_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos));

    this->temp1 = vectorfield( this->nos, {0, 0, 0} );
    this->temp2 = vectorfield( this->nos, {0, 0, 0} );
    this->spins_predictor = vectorfield( this->nos, {0, 0, 0} );
  }
  
  // void Optimizer_Heun::Iteration()
  // {
  //   // Get the actual forces on the configurations
  //   this->method->Calculate_Force( this->configurations, this->force );
  //   
  //   // Optimization for each image
  //   for (int i = 0; i < this->noi; ++i)
  //   {
  //     this->s = method->systems[i];
  //     auto& conf = *this->configurations[i];
  //     
  //     // Scaling to similar stability range as SIB
  //     this->dt = this->s->llg_parameters->dt/100;
  //     
  //     // TODO: for dynamic solver we would need a damping factor alpha
  //     
  //     // First step - Predictor
  //     Vectormath::set_c_cross( 1, conf, force[i], temp2 );    // temp2 = conf x force
  //     Vectormath::set_c_cross( -dt, conf, temp2, temp1 );     // temp1 = -dt * conf x temp2
  //     Vectormath::set_c_a( 1, conf, spins_predictor );        // spins_predictor = conf
  //     Vectormath::add_c_a( 1, temp1, spins_predictor );       // spins_predictor += temp1
  //     
  //     // Second step - Corrector
  //     Vectormath::set_c_cross( 1, spins_predictor, force[i], temp2 );   // temp2 += spins_predictor x force
  //     Vectormath::add_c_cross( -dt, spins_predictor, temp2, temp1 );    // temp1 += -dt*spins_predictor x temp2
  //     Vectormath::set_c_a( 0.5*dt, conf, spins_temp[i] );               // spins_temp = 0.5*dt*conf
  //     Vectormath::add_c_a( 0.5*dt, temp1, spins_temp[i] );              // spins_temp += 0.5*dt*temp1
  //     
  //     // Normalize spins
  //     Vectormath::normalize_vectors( spins_temp[i] );
  //     
  //     // Copy out
  //     conf = spins_temp[i];
  //   } 
  // }

  void Optimizer_Heun::Iteration()
  {    
    
    // Get the actual forces on the configurations
    this->method->Calculate_Force( this->configurations, this->force );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
      this->s = method->systems[i];
      auto& conf = *this->configurations[i];
      
      // First step - Predictor
      this->VirtualForce( *s->spins, *s->llg_parameters, force[i], virtualforce[i] );
      Vectormath::set_c_cross( -1, conf, virtualforce[i], temp1 );  // temp1 = -( conf x A )
      Vectormath::set_c_a( 1, conf, spins_predictor );              // spins_predictor = conf
      Vectormath::add_c_a( 1, temp1, spins_predictor );            // spins_predictor = conf + dt*temp1
      
      // Normalize spins
      Vectormath::normalize_vectors( spins_predictor );
      
      *spins_temp[i] = spins_predictor;
    }
    
    this->method->Calculate_Force( spins_temp, this->force );
    
    for (int i=0; i < this->noi; i++)
    {
      this->s = method->systems[i];
      auto& conf = *this->configurations[i];

      //Second step - Corrector
      this->VirtualForce( spins_predictor, *s->llg_parameters, force[i], virtualforce[i] );
      
      Vectormath::set_c_a( 1, conf, *spins_temp[i] );         // spins_temp = conf
      Vectormath::add_c_a( 0.5, temp1, *spins_temp[i] );   // spins_temp = conf + 0.5 * dt* temp1 
      Vectormath::set_c_cross( -1, spins_predictor, virtualforce[i], temp1 );  // temp1' = - ( conf' x A' )
      Vectormath::add_c_a( 0.5, temp1, *spins_temp[i] );   // spins_temp = conf + 0.5 * dt * temp1 + 0.5 * dt * temp1'        

      // Normalize spins
      Vectormath::normalize_vectors( *spins_temp[i] );
      
      // Copy out
      conf = *spins_temp[i];
    } 
  }

  void Optimizer_Heun::VirtualForce( const vectorfield & spins, 
                                     const Data::Parameters_Method_LLG & llg_params, 
                                     const vectorfield & effective_field, 
                                     vectorfield & virtualforce )
  {
    //========================= Init local vars ================================
    // time steps
    scalar damping = llg_params.damping;
    // dt = time_step [ps] * 10^-12 * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
    scalar dtg = llg_params.dt * 1e-12 * Constants::gamma / Constants::mu_B / (1 + damping*damping);
    //scalar sqrtdtg = dtg / std::sqrt( llg_params.dt );
    // STT
    scalar a_j = llg_params.stt_magnitude;
    Vector3 s_c_vec = llg_params.stt_polarisation_normal;
    //------------------------ End Init ----------------------------------------
    
    Vectormath::fill( virtualforce, { 0, 0, 0 } );
    //Vectormath::add_c_a( -0.5 * dtg, effective_field, virtualforce );
    Vectormath::set_c_cross( dtg, spins, effective_field, virtualforce );
    
    // Apply Pinning
    #ifdef SPIRIT_ENABLE_PINNING
        Vectormath::set_c_a( 1, force, force, llg_params.pinning->mask_unpinned );
    #endif // SPIRIT_ENABLE_PINNING
  }

  // Optimizer name as string
  std::string Optimizer_Heun::Name() { return "Heun"; }
  std::string Optimizer_Heun::FullName() { return "Heun"; }
}