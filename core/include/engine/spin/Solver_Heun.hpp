#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::Heun> : public SolverMethods
{
protected:
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::SolverMethods;
    // Actual Forces on the configurations
    std::vector<vectorfield> forces_predictor;
    // Virtual Forces used in the Steps
    std::vector<vectorfield> forces_virtual_predictor;

    std::vector<std::shared_ptr<vectorfield>> configurations_predictor;
    std::vector<std::shared_ptr<vectorfield>> delta_configurations;
};

template<>
inline void Method_Solver<Solver::Heun>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::make_shared<vectorfield>( this->nos );

    this->delta_configurations = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        delta_configurations[i] = std::make_shared<vectorfield>( this->nos );
}

/*
    Template instantiation of the Simulation class for use with the Heun Solver.
        The Heun method is a basic solver for the PDE at hand here. It is sufficiently
        efficient and stable.
    This method is described for spin systems including thermal noise in
        U. Nowak, Thermally Activated Reversal in Magnetic Nanostructures,
        Annual Reviews of Computational Physics IX Chapter III (p 105) (2001)
*/
template<>
inline void Method_Solver<Solver::Heun>::Iteration()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        // First step - Predictor
        Solver_Kernels::heun_predictor(
            *this->configurations[i], this->forces_virtual[i], *this->delta_configurations[i],
            *this->configurations_predictor[i] );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int i = 0; i < this->noi; i++ )
    {
        // Second Step - Corrector
        Solver_Kernels::heun_corrector(
            this->forces_virtual_predictor[i], *this->delta_configurations[i],
            *this->configurations_predictor[i], *this->configurations[i] );
    }
}

} // namespace Spin

} // namespace Engine
