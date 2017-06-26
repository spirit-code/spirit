#include <engine/Optimizer_NCG.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Hamiltonian.hpp>

namespace Engine
{
  Optimizer_NCG::Optimizer_NCG( std::shared_ptr<Engine::Method> method ):
    Optimizer( method )
  {
    this->jmax    = 500;    // max iterations
    this->tol_NR  = 1e-5;   // Newton-Raphson error tolerance
    this->tol_nGB = 1e-5;   // XXX: should I provide the tolerance for the optimizer ?
    
    this->method->Calculate_Force( this->configurations, this->force );   // initialize residual
    
    for (int img=0; img<this->noi; img++)
    {
      Engine::Vectormath::set_c_a( -1, this->force[img], this->res[img] );    // set residual
      
      this->delta_new[img] = Engine::Vectormath::dot( this->res[img], this->res[img] ); // delta new
      
      Engine::Vectormath::set_c_a( 1, this->res[img], this->d[img] );   // set d
    }
    
    // XXX: is the assignment operator working properly for the scalarfields?
    this->delta_0  = this->delta_new;    // save initial delta new
    this->beta  = 0;  // set beta to zero
    this->alpha = 0;  // set alpha to zero
    //+ beta*d
  }
  
  void Optimizer_NCG::Iteration()
  {
    int convergence;                      // convergence state
    double eps = std::pow( tol_NR, 2 );   // calculate NR convergence criterion
    
    // calculate delta_d
    for (int img=0; img<this->noi; img++)
      this->delta_d[img] = Engine::Vectormath::dot( this->d[img], this->d[img] );  
    
    // Perform a Newton-Raphson line search in order to determine the minimum along d  
    do
    {
      convergence = 1;                  // set convergence to true
      
      // alpha = - (f'*d)/(d*f''*d)	// TODO: How to get the second derivative from here??
      //this->alpha = - 
      //this->method->Calculate( )
      
      //this->alpha = - Engine::Vectormath::dot( )
      
      //this->x += a * this->d; 
      // x = x + alpha*d
      
      // check convergence
      for ( int img=0; img<this->noi; img++)
        convergence *= ( this->delta_d[img]*this->alpha > eps ) ? 0 : 1; 
    }
    while (int j=0; j< jmax && convergence; j++ )

    
    // Update the direction d
    // r = -f' = eff_field
    // delta_old = delta_new
    // delta_new = r*r
    // beta = delta_new/delta_old
    // d = r + beta*d
  
    // Restart if d is not a descent direction or after nos iterations
    //		The latter improves convergence for small nos
    // ++k
    // if (r*d <= 0 || k==nos)
    //{
    // d = r
      // k = 0
    //}
  }
  
  // Optimizer name as string
  std::string Optimizer_NCG::Name() { return "NCG"; }
  std::string Optimizer_NCG::FullName() { return "Nonlinear Conjugate Gradient"; }
}