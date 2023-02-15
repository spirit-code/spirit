#include "engine/Manifoldmath.hpp"
#include "engine/Solver_Kernels.hpp"
#include <engine/Backend_par.hpp>

template<>
inline void Method_Solver<Solver::SIB>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );
}

/*
    Template instantiation of the Simulation class for use with the SIB Solver.
        The semi-implicit method B is an efficient midpoint solver.
    Paper: J. H. Mentink et al., Stable and fast semi-implicit integration of the stochastic
           Landau-Lifshitz equation, J. Phys. Condens. Matter 22, 176001 (2010).
*/
template<>
inline void Method_Solver<Solver::SIB>::Iteration()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // First part of the step
    // Calculate forces for current configuration
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        // clang-format off
        Backend::par::apply( nos, 
            [
                c = this->configurations[i]->data(),
                c_p = this->configurations_predictor[i]->data(),
                f = forces_virtual[i].data()
            ] SPIRIT_LAMBDA (int idx)
            {
                c_p[idx] = Engine::Solver_Kernels::cayley_transform(0.5 * f[idx], c[idx]);
                c_p[idx] = 0.5 * (c_p[idx] + c[idx]);
            } 
        );
        // clang-format on
    }

    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    for( int i = 0; i < this->noi; ++i )
    {
        // clang-format off
        Backend::par::apply( nos, 
            [
                c = this->configurations[i]->data(),
                c_p = this->configurations_predictor[i]->data(),
                f_p = forces_virtual_predictor[i].data()
            ] SPIRIT_LAMBDA (int idx)
            {
                f_p[idx] = f_p[idx] - f_p[idx].dot(c_p[idx]) * c_p[idx]; // Remove normal component of predictor
                c_p[idx] = Engine::Solver_Kernels::cayley_transform(0.5 * f_p[idx], c[idx]);
                c[idx] = c_p[idx];
            } 
        );
        // clang-format on
    }
}

template<>
inline std::string Method_Solver<Solver::SIB>::SolverName()
{
    return "SIB";
}

template<>
inline std::string Method_Solver<Solver::SIB>::SolverFullName()
{
    return "Semi-implicit B";
}