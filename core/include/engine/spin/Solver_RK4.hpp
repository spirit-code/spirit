#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::RungeKutta4> : public SolverMethods
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

    std::vector<std::shared_ptr<vectorfield>> configurations_k1;
    std::vector<std::shared_ptr<vectorfield>> configurations_k2;
    std::vector<std::shared_ptr<vectorfield>> configurations_k3;
};

template<>
inline void Method_Solver<Solver::RungeKutta4>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_predictor = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_predictor[i] = std::make_shared<vectorfield>( this->nos );

    this->configurations_k1 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k1[i] = std::make_shared<vectorfield>( this->nos );

    this->configurations_k2 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k2[i] = std::make_shared<vectorfield>( this->nos );

    this->configurations_k3 = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        this->configurations_k3[i] = std::make_shared<vectorfield>( this->nos );
}

/*
    Template instantiation of the Simulation class for use with the 4th order Runge Kutta Solver.
*/
template<>
inline void Method_Solver<Solver::RungeKutta4>::Iteration()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // Get the actual forces on the configurations
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        Solver_Kernels::rk4_predictor_1(
            *this->configurations[i], this->forces_virtual[i], *this->configurations_k1[i],
            *this->configurations_predictor[i] );
    }

    // Calculate_Force for the predictor
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        Solver_Kernels::rk4_predictor_2(
            *this->configurations[i], this->forces_virtual_predictor[i], *this->configurations_k2[i],
            *this->configurations_predictor[i] );
    }

    // Calculate_Force for the predictor (k3)
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        Solver_Kernels::rk4_predictor_3(
            *this->configurations[i], this->forces_virtual_predictor[i], *this->configurations_k3[i],
            *this->configurations_predictor[i] );
    }

    // Calculate_Force for the predictor (k4)
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Corrector step for each image
    for( int i = 0; i < this->noi; i++ )
    {
        Solver_Kernels::rk4_corrector(
            forces_virtual_predictor[i], *configurations_k1[i], *configurations_k2[i], *configurations_k3[i],
            *configurations_predictor[i], *configurations[i] );
    }
}

} // namespace Spin

} // namespace Engine
