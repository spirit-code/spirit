#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::Depondt> : public SolverMethods
{
protected:
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::SolverMethods;
    // Temporaries for virtual forces
    std::vector<vectorfield> rotationaxis;
    std::vector<scalarfield> forces_virtual_norm;
    // Preccession angle
    scalarfield angle;

    // Actual Forces on the configurations
    std::vector<vectorfield> forces_predictor;
    // Virtual Forces used in the Steps
    std::vector<vectorfield> forces_virtual_predictor;

    std::vector<std::shared_ptr<vectorfield>> configurations_predictor;

    vectorfield temp1;
};

template<>
inline void Method_Solver<Solver::Depondt>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->rotationaxis        = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->angle               = scalarfield( this->nos, 0 );
    this->forces_virtual_norm = std::vector<scalarfield>( this->noi, scalarfield( this->nos, 0 ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::make_shared<vectorfield>( this->nos, Vector3{ 0, 0, 0 } );

    this->temp1 = vectorfield( this->nos, { 0, 0, 0 } );
}

/*
    Template instantiation of the Simulation class for use with the Depondt Solver.
        The Depondt method is an improvement of Heun's method for spin systems. It applies
        rotations instead of finite displacements and thus avoids re-normalizations.
    Paper: Ph. Depondt et al., Spin dynamics simulations of two-dimensional clusters with
           Heisenberg and dipole-dipole interactions, J. Phys. Condens. Matter 21, 336005 (2009).
*/
template<>
inline void Method_Solver<Solver::Depondt>::Iteration()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        Solver_Kernels::depondt_predictor(
            forces_virtual[i], rotationaxis[i], angle, *configurations[i], *configurations_predictor[i] );
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int i = 0; i < this->noi; i++ )
    {
        Solver_Kernels::depondt_corrector(
            forces_virtual[i], forces_virtual_predictor[i], temp1, angle, *configurations[i] );
    }
}

} // namespace Spin

} // namespace Engine
