#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::VP_OSO> : public SolverMethods
{
protected:
    using SolverMethods::SolverMethods;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;

    // Force in previous step [noi][nos]
    std::vector<vectorfield> forces_previous;
    // Velocity used in the Steps [noi][nos]
    std::vector<vectorfield> velocities;

    std::vector<vectorfield> grad;
    std::vector<vectorfield> grad_pr;
    std::vector<vectorfield> searchdir;

    std::vector<std::shared_ptr<vectorfield>> configurations_temp;

    std::vector<std::shared_ptr<const Data::Parameters_Method_LLG>> llg_parameters;
};

template<>
inline void Method_Solver<Solver::VP_OSO>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::make_shared<vectorfield>( this->nos );

    this->velocities = std::vector<vectorfield>( this->noi, vectorfield( this->nos, Vector3::Zero() ) ); // [noi][nos]
    this->forces_previous     = forces;                                                                  // [noi][nos]
    this->grad                = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr             = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->searchdir           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

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

    Instead of the cartesian update scheme with re-normalization, this implementation uses the orthogonal spin
   optimization scheme, described by A. Ivanov in https://arxiv.org/abs/1904.02669.
*/

template<>
inline void Method_Solver<Solver::VP_OSO>::Iteration()
{
    // Set previous
    for( int img = 0; img < noi; ++img )
    {
        Backend::copy( SPIRIT_PAR grad[img].begin(), grad[img].end(), grad_pr[img].begin() );
    }

    // Get the forces on the configurations
    this->Calculate_Force( configurations, forces );
    this->Calculate_Force_Virtual( configurations, forces, forces_virtual );

    for( int img = 0; img < this->noi; img++ )
    {
        auto & image = *this->configurations[img];
        auto & grad  = this->grad[img];
        Solver_Kernels::oso_calc_gradients( grad, image, this->forces[img] );
        Vectormath::scale( grad, -1.0 );

        Solver_Kernels::VP::bare_velocity( this->grad[img], this->grad_pr[img], velocities[img] );
    }

    // Get the total projection of the velocity on the force
    const Vector2 projections = Backend::cpu::transform_reduce(
        velocities.begin(), velocities.end(), grad.begin(), Vector2{ 0.0, 0.0 }, Backend::plus<Vector2>{},
        []( const vectorfield & velocity, const vectorfield & force ) -> Vector2
        { return Vector2{ Vectormath::dot( velocity, force ), Vectormath::dot( force, force ) }; } );

    for( int img = 0; img < noi; ++img )
    {
        // Calculate the projected velocity
        Solver_Kernels::VP::projected_velocity( projections, grad[img], velocities[img] );

        // Apply the projected velocity
        Solver_Kernels::VP::set_step( velocities[img], grad[img], this->llg_parameters[img]->dt, searchdir[img] );

        // rotate spins
        Solver_Kernels::oso_rotate( *this->configurations[img], this->searchdir[img] );
    }
}

} // namespace Spin

} // namespace Engine
