#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::VP> : public SolverMethods
{
protected:
    using SolverMethods::SolverMethods;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    //////////// VP ///////////////////////////////////////////////////////////////

    // Force in previous step [noi][nos]
    std::vector<vectorfield> forces_previous;
    // Velocity used in the Steps [noi][nos]
    std::vector<vectorfield> velocities;

    std::vector<std::shared_ptr<const Data::Parameters_Method_LLG>> llg_parameters;
};

template<>
inline void Method_Solver<Solver::VP>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->forces_previous = forces;                                                                      // [noi][nos]

    this->llg_parameters = std::vector<std::shared_ptr<const Data::Parameters_Method_LLG>>( this->noi, nullptr );
    for( int i = 0; i < this->noi; i++ )
        this->llg_parameters[i] = this->systems[i]->llg_parameters;
}

/*
    Template instantiation of the Simulation class for use with the VP Solver.
        The velocity projection method is often efficient for direct minimization,
        but deals poorly with quickly varying fields or stochastic noise.
    Paper: P. F. Bessarab et al., Method for finding mechanism and activation energy
           of magnetic transitions, applied to skyrmion and antivortex annihilation,
           Comp. Phys. Comm. 196, 335 (2015).
*/
template<>
inline void Method_Solver<Solver::VP>::Iteration()
{
    // Set previous
    for( int i = 0; i < noi; ++i )
    {
        Backend::copy( SPIRIT_PAR forces[i].begin(), forces[i].end(), forces_previous[i].begin() );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int i = 0; i < noi; ++i )
    {
        Solver_Kernels::VP::bare_velocity( forces[i], forces_previous[i], velocities[i] );
    }

    // Get the projection of the velocity on the force
    const Vector2 projections = Backend::cpu::transform_reduce(
        velocities.begin(), velocities.end(), forces.begin(), Vector2{ 0.0, 0.0 }, Backend::plus<Vector2>{},
        []( const vectorfield & velocity, const vectorfield & force ) -> Vector2
        { return Vector2{ Vectormath::dot( velocity, force ), Vectormath::dot( force, force ) }; } );

    for( int i = 0; i < noi; ++i )
    {
        // Calculate the projected velocity
        Solver_Kernels::VP::projected_velocity( projections, forces[i], velocities[i] );

        // Apply the projected velocity
        Solver_Kernels::VP::apply_velocity( velocities[i], forces[i], this->llg_parameters[i]->dt, *configurations[i] );
    }
}

} // namespace Spin

} // namespace Engine
