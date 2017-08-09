#include <engine/Optimizer_Depondt.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Dense>

using namespace Utility;

namespace Engine
{
  Optimizer_Depondt::Optimizer_Depondt( std::shared_ptr<Engine::Method> method ) :
        Optimizer( method )
  {

    this->virtualforce = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->rotationaxis = std::vector<vectorfield>( this->noi, vectorfield( this->nos, {0, 0, 0} ) );
    this->virtualforce_predictor = std::vector<vectorfield>( this->noi, 
                                                             vectorfield( this->nos, {0, 0, 0} ) );
    
    this->spins_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      spins_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
    
    this->spins_temp  = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
      spins_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
        
    this->angle = scalarfield( this->nos, 0 );
    this->virtualforce_norm = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );  
  }

  void Optimizer_Depondt::Iteration()
  {
    // Get the actual forces on the configurations
    this->method->Calculate_Force( this->configurations, this->force );
    
    // Optimization for each image
    for (int i = 0; i < this->noi; ++i)
    {
      this->s = method->systems[i];
      auto& conf = *this->configurations[i];
      this->dtg = this->s->llg_parameters->dt * Constants::gamma / Constants::mu_B / 
                  ( 1 + pow( this->s->llg_parameters->damping, 2 )  );
      
      // Calculate Virtual force H
      this->VirtualForce( *s->spins, *s->llg_parameters, force[i], virtualforce[i] );
      
      // For Rotation matrix R := R( H_normed, angle )
      Vectormath::norm( virtualforce[i], angle );   // angle = |virtualforce|
    
      Vectormath::set_c_a( 1, virtualforce[i], rotationaxis[i] );  // rotationaxis = |virtualforce|
      Vectormath::normalize_vectors( rotationaxis[i] );            // normalize rotation axis 
      
      Vectormath::scale( angle, dtg );    // angle = |virtualforce| * dt
      
      // Get spin predictor n' = R(H) * n
      Vectormath::rotate( conf, rotationaxis[i], angle, *spins_predictor[i] );  
    }
    
    this->method->Calculate_Force( spins_predictor, this->force );
    
    for (int i=0; i < this->noi; i++)
    {
      this->s = method->systems[i];
      auto& conf = *this->configurations[i];
      this->dtg = this->s->llg_parameters->dt * Constants::gamma / Constants::mu_B / 
                  ( 1 + pow( this->s->llg_parameters->damping, 2 )  );
      
      // Calculate Predicted Virtual force H'
      this->VirtualForce( *spins_predictor[i], *s->llg_parameters, force[i], 
                          virtualforce_predictor[i] );
      
      // Calculate the linear combination of the two virtualforces
      Vectormath::scale( virtualforce[i], 0.5 );   // H = H/2
      Vectormath::add_c_a( 0.5, virtualforce_predictor[i], virtualforce[i] ); // H = (H + H')/2
      
      // For Rotation matrix R' := R( H'_normed, angle' )
      Vectormath::norm( virtualforce[i], angle );   // angle' = |virtualforce lin combination|
      Vectormath::scale( angle, dtg );              // angle' = |virtualforce lin combination| * dt
      
      Vectormath::normalize_vectors( virtualforce[i] );  // normalize virtual force
        
      // Get new spin conf n_new = R( (H+H')/2 ) * n
      Vectormath::rotate( conf, virtualforce[i], angle, conf );  
    }
  }
  
  void Optimizer_Depondt::VirtualForce( const vectorfield & spins, 
                                        const Data::Parameters_Method_LLG & llg_params, 
                                        const vectorfield & effective_field, 
                                        vectorfield & virtualforce )
  {
    //========================= Init local vars ================================
    // time steps
    scalar damping = llg_params.damping;
    // dt = time_step [ps] * 10^-12 * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
    //scalar dtg = llg_params.dt * 1e-12 * Constants::gamma / Constants::mu_B / (1 + damping*damping);
    //scalar sqrtdtg = dtg / std::sqrt( llg_params.dt );
    // STT
    scalar a_j = llg_params.stt_magnitude;
    Vector3 s_c_vec = llg_params.stt_polarisation_normal;
    //------------------------ End Init ----------------------------------------
    
    Vectormath::fill( virtualforce, { 0, 0, 0 } );
    Vectormath::set_c_cross( 1, spins, effective_field, virtualforce );
    
    // Apply Pinning
    #ifdef SPIRIT_ENABLE_PINNING
        Vectormath::set_c_a( 1, force, force, llg_params.pinning->mask_unpinned );
    #endif // SPIRIT_ENABLE_PINNING
  }
  
  // Optimizer's name as string
  std::string Optimizer_Depondt::Name() { return "Depondt"; }
  std::string Optimizer_Depondt::FullName() { return "Depondt"; }
}
