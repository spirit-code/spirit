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
    // "Mass of our particle" which we accelerate
    static constexpr scalar mass = 1.0;

    // Force in previous step [noi][nos]
    std::vector<vectorfield> forces_previous;
    // Velocity in previous step [noi][nos]
    std::vector<vectorfield> velocities_previous;
    // Velocity used in the Steps [noi][nos]
    std::vector<vectorfield> velocities;
    // Projection of velocities onto the forces [noi]
    std::vector<scalar> projection;
    // |force|^2
    std::vector<scalar> force_norm2;

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
    this->velocities_previous = velocities;                                                              // [noi][nos]
    this->forces_previous     = velocities;                                                              // [noi][nos]
    this->grad                = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr             = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->projection          = std::vector<scalar>( this->noi, 0 ); // [noi]
    this->force_norm2         = std::vector<scalar>( this->noi, 0 ); // [noi]
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
    scalar projection_full  = 0;
    scalar force_norm2_full = 0;

    // Set previous
    for( int img = 0; img < noi; ++img )
    {
        const auto * g = raw_pointer_cast( grad[img].data() );
        const auto * v = raw_pointer_cast( velocities[img].data() );
        auto * g_pr    = raw_pointer_cast( grad_pr[img].data() );
        auto * v_pr    = raw_pointer_cast( velocities_previous[img].data() );

        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), nos,
            [g, g_pr, v, v_pr] SPIRIT_LAMBDA( const int idx )
            {
                g_pr[idx] = g[idx];
                v_pr[idx] = v[idx];
            } );
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
    }

    for( int img = 0; img < noi; ++img )
    {
        const auto * g    = raw_pointer_cast( this->grad[img].data() );
        const auto * g_pr = raw_pointer_cast( this->grad_pr[img].data() );
        auto & velocity   = velocities[img];
        auto * v          = raw_pointer_cast( velocities[img].data() );

        // Calculate the new velocity
        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), nos,
            [g, g_pr, v] SPIRIT_LAMBDA( const int idx )
            { v[idx] += 0.5 / mass * ( g_pr[idx] + g[idx] ); } );

        // Get the projection of the velocity on the force
        projection[img]  = Vectormath::dot( velocity, this->grad[img] );
        force_norm2[img] = Vectormath::dot( this->grad[img], this->grad[img] );
    }
    for( int img = 0; img < noi; ++img )
    {
        projection_full += projection[img];
        force_norm2_full += force_norm2[img];
    }
    for( int img = 0; img < noi; ++img )
    {
        const auto * g = raw_pointer_cast( this->grad[img].data() );
        auto * sd      = raw_pointer_cast( this->searchdir[img].data() );
        auto * v       = raw_pointer_cast( this->velocities[img].data() );

        scalar dt    = this->llg_parameters[img]->dt;
        scalar ratio = projection_full / force_norm2_full;

        // Calculate the projected velocity
        if( projection_full <= 0 )
        {
            Vectormath::fill( velocities[img], { 0, 0, 0 } );
        }
        else
        {
            Backend::for_each_n(
                SPIRIT_PAR Backend::make_counting_iterator( 0 ), nos,
                [g, v, ratio] SPIRIT_LAMBDA( const int idx ) { v[idx] = g[idx] * ratio; } );
        }

        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), nos,
            [sd, dt, v, g] SPIRIT_LAMBDA( const int idx )
            { sd[idx] = dt * v[idx] + 0.5 / mass * dt * g[idx]; } );
    }
    Solver_Kernels::oso_rotate( this->configurations, this->searchdir );
}

template<>
inline std::string Method_Solver<Solver::VP_OSO>::SolverName()
{
    return "VP_OSO";
}

template<>
inline std::string Method_Solver<Solver::VP_OSO>::SolverFullName()
{
    return "Velocity Projection using exponential transforms";
}

} // namespace Spin

} // namespace Engine
