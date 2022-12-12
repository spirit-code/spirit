#include "data/Parameters_Method_LLG.hpp"
#include "data/Spin_System.hpp"
// #include "engine/Hamiltonian_Heisenberg.hpp"
#include "engine/Solver_Kernels.hpp"
#include "engine/Vectormath_Defines.hpp"
#include <fmt/format.h>

#include <limits>

using ST_Propagator = Data::Definitions::ST_Propagator;

template<>
inline void Method_Solver<Solver::ST>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]

    const Data::Parameters_Method_LLG * parameters = (const Data::Parameters_Method_LLG *)this->Parameters();
    this->st_propagator                            = parameters->st_propagator;

    auto & ham                          = *this->System( 0 )->hamiltonian;
    this->has_linear_self_contributions = ham.Has_Linear_Self_Contributions();
}

inline Vector3 Force_Single_Spin(
    int ispin, const Vector3 & gradient, const vectorfield & spins, const Method_Solver<Solver::ST> & solver )
{
    const Data::Parameters_Method_LLG * parameters = (const Data::Parameters_Method_LLG *)solver.Parameters();
    // time steps
    scalar damping = parameters->damping;
    // dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
    scalar dtg = parameters->dt * Constants::gamma / Constants::mu_B / ( 1 + damping * damping );

    const Vector3 force         = gradient; // actually it should be -gradient
    const Vector3 force_virtual = ( dtg * force + dtg * damping * spins[ispin].cross( force ) );

    if( parameters->time_reversal )
        return -force_virtual;

    return force_virtual;
}

inline Vector3 analytical(
    scalar time_step_factor, Vector3 & spin_initial, const scalar field_z, const Method_Solver<Solver::ST> & solver )
{
    const Data::Parameters_Method_LLG * parameters = (const Data::Parameters_Method_LLG *)solver.Parameters();

    // time steps
    scalar alpha = parameters->damping;
    scalar gamma = Constants::gamma / Constants::mu_B;
    scalar dt    = parameters->dt * time_step_factor;

    const scalar theta0 = std::acos( spin_initial[2] );
    const scalar phi0   = std::atan2( spin_initial[1], spin_initial[0] );
    const scalar theta
        = 2.0
          * std::atan( std::tan( theta0 / 2.0 ) * std::exp( -gamma * field_z * dt * alpha / ( 1.0 + alpha * alpha ) ) );

    const scalar phi = phi0 + gamma / ( 1.0 + alpha * alpha ) * field_z * dt;

    // std::cout << "phi " << phi << "\n";
    // std::cout << "theta " << theta << "\n";

    if( std::isnan( theta ) )
        return spin_initial;

    return Vector3{ std::cos( phi ) * std::sin( theta ), std::sin( phi ) * std::sin( theta ), std::cos( theta ) };
}

inline scalar Angle( const Vector3 & a, const Vector3 & b )
{
    // vincentys formula
    const scalar angle = std::atan2( a.cross( b ).norm(), a.dot( b ) );
    if( std::isnan( angle ) )
        return 0.0;
    return angle;
}

inline void SA_Propagator(
    scalar time_scale, const Vector3 & gradient, int ispin, vectorfield & spins,
    const Method_Solver<Solver::ST> & solver )
{

    const Data::Parameters_Method_LLG * parameters = (const Data::Parameters_Method_LLG *)solver.Parameters();

    const Vector3 ez = Vector3{ 0, 0, 1 };

    // Rotate into a frame where the gradient (not the force!) points into z direction
    const scalar angle = Angle( gradient, ez );
    const Vector3 axis = gradient.cross( ez );

    const Matrix3 rotation_matrix = Eigen::AngleAxis<scalar>( angle, axis.normalized() ).toRotationMatrix();

    Vector3 spin_temp = rotation_matrix * spins[ispin];

    if( parameters->time_reversal )
        time_scale *= -1;

    spin_temp = rotation_matrix.transpose() * analytical( time_scale, spin_temp, gradient.norm(), solver );

    spins[ispin] = spin_temp;
    spins[ispin].normalize();
}

inline void
Heun_Propagator( scalar time_scale, int ispin, vectorfield & spins, const Method_Solver<Solver::ST> & solver )
{
    Vector3 spin_temp = spins[ispin];
    Vector3 gradient  = -solver.System( 0 )->hamiltonian->Gradient_Single_Spin( ispin, spins );
    Vector3 f         = Force_Single_Spin( ispin, gradient, spins, solver );

    // Predictor force
    spins[ispin] -= time_scale * spins[ispin].cross( f );
    spins[ispin].normalize();

    gradient    = -solver.System( 0 )->hamiltonian->Gradient_Single_Spin( ispin, spins );
    Vector3 f_p = Force_Single_Spin( ispin, gradient, spins, solver );

    f_p = f_p - f_p.dot( spins[ispin] ) * spins[ispin];
    f_p = f_p - f_p.dot( spin_temp ) * spin_temp;

    // Restore configuration
    spins[ispin] = spin_temp;

    Vector3 avg_force = 0.5 * ( f + f_p );
    spins[ispin] -= time_scale * spins[ispin].cross( avg_force );
    spins[ispin].normalize();
}

template<typename Force_Callback>
inline void Implicit_Propagator(
    scalar time_scale, int ispin, vectorfield & spins, Force_Callback & force_callback,
    const Method_Solver<Solver::ST> & solver )
{
    // Solve the equation s_{i+1} = s_{i} + 0.5 * (s_{i} + s_{i+1}) x 0.5 * ( f(s_{i+1}) + f(s_i)) ), for a single spin
    const scalar convergence = 1e-16;
    const int max_iter       = 200;

    const Vector3 spin_initial = spins[ispin];
    Vector3 spin_previous      = spins[ispin];
    Vector3 spin_propagated    = spins[ispin];
    Vector3 spin_avg           = spins[ispin];

    // Vector3 force         = force_callback( ispin, spins );
    // Vector3 force_initial = force; // Initialize to current force, so that first iteration performs one SiB update
    Vector3 force_avg = force_callback( ispin, spins );

    int iter = 0;
    bool run = true;

    while( run )
    {
        // Save the current spin
        spin_previous = spin_propagated;

        // Compute the propagated spin
        // Possibility1: naive
        spin_propagated = spin_initial - spin_avg.cross( force_avg );

        // Possibility2: cayley
        // spin_propagated = Engine::Solver_Kernels::cayley_transform( 0.5 * force_avg, spin_initial );

        // Compute the average spin
        spin_avg = 0.5 * ( spin_propagated + spin_initial );

        // Compute the average force
        // f( spin_avg )
        spins[ispin] = spin_avg;
        force_avg    = time_scale * force_callback( ispin, spins );

        iter++;
        scalar change = ( spin_propagated - spin_previous ).cwiseAbs().maxCoeff();

        run = change > convergence && iter < max_iter;
    }

    // Assign the propagated spin to the spins array
    spins[ispin] = spin_propagated;
}

template<typename Gradient_Callback>
inline void SA_Implicit_Propagator(
    scalar time_scale, int ispin, vectorfield & spins, Gradient_Callback & gradient_callback,
    const Method_Solver<Solver::ST> & solver )
{
    scalar convergence = 1e-16;
    int max_iter       = 200;

    Vector3 spin_initial    = spins[ispin];
    Vector3 spin_previous   = spins[ispin];
    Vector3 spin_propagated = spins[ispin];
    Vector3 spin_avg        = spins[ispin];

    Vector3 gradient         = gradient_callback( ispin, spins );
    Vector3 gradient_initial = gradient; // Initialize to current force, so that first iteration performs one SiB update
    Vector3 gradient_avg     = gradient;

    int iter = 0;
    bool run = true;

    while( run )
    {
        // Save the current spin
        spin_previous = spin_propagated;

        // Compute the propagated spin
        spins[ispin] = spin_initial;
        SA_Propagator( time_scale, gradient_avg, ispin, spins, solver );
        spin_propagated = spins[ispin];

        // Compute the average spin
        spin_avg = 0.5 * ( spin_propagated + spin_initial );

        // Compute the average force
        //  Possibility1: f( spin_avg )
        spins[ispin] = spin_avg;
        gradient_avg = time_scale * gradient_callback( ispin, spins );

        //  Possibility2: 0.5 * (f( s1 ) + f( s2 ) )
        // spins[ispin] = spin_propagated;
        // force_avg    = time_scale * 0.5 * ( force_callback(ispin, spins)  + force_initial );
        // force_avg = force_avg - force_avg.dot(spin_avg.normalized()) * spin_avg.normalized();

        iter++;
        scalar change = ( spin_propagated - spin_previous ).cwiseAbs().maxCoeff();

        run = change > convergence && iter < max_iter;
        // std::cout << iter << " " << change << "\n";
    }

    // Assign the propagated spin to the spins array
    spins[ispin] = spin_propagated;

    // std::cout << iter << "\n";
}

template<>
inline void Method_Solver<Solver::ST>::Iteration()
{
    // const Data::Parameters_Method_LLG::ST
    const Data::Parameters_Method_LLG * parameters = (const Data::Parameters_Method_LLG *)this->Parameters();

    for( int img = 0; img < this->noi; ++img )
    {
        auto & spins = *this->systems[img]->spins;

        // Compute this information so that we do not have to re-compute the entire gradient in every iteration
        Vector3 gradient_current;
        Matrix3 gradient_linear_current;
        Vector3 spin_current;

        const auto force_callback = [&solver = *this, &gradient_current, &gradient_linear_current,
                                     &spin_current]( int ispin, const vectorfield & spins ) {
            Vector3 gradient = -( gradient_current + gradient_linear_current * ( spins[ispin] - spin_current ) );
            Vector3 f        = Force_Single_Spin( ispin, gradient, spins, solver );
            return f;
        };

        const auto gradient_callback = [&solver = *this, &gradient_current, &gradient_linear_current,
                                        &spin_current]( int ispin, const vectorfield & spins ) {
            Vector3 gradient = -( gradient_current + gradient_linear_current * ( spins[ispin] - spin_current ) );
            return gradient;
        };

        for( int ispin = 0; ispin < nos; ispin++ )
        {
            gradient_current = this->System( 0 )->hamiltonian->Gradient_Single_Spin( ispin, spins );
            gradient_linear_current
                = this->System( 0 )->hamiltonian->Linear_Gradient_Contribution_Single_Spin( ispin, spins );
            spin_current = spins[ispin];

            if( this->st_propagator == ST_Propagator::IMP )
            {
                Implicit_Propagator( 0.5, ispin, spins, force_callback, *this );
            }
            else
            {
                if( this->has_linear_self_contributions )
                {
                    SA_Implicit_Propagator( 0.5, ispin, spins, gradient_callback, *this );
                }
                else
                {
                    SA_Propagator( 0.5, -gradient_current, ispin, spins, *this );
                }
            }
            // Vector3 gradient           = -this->systems[img]->hamiltonian->Gradient_Single_Spin( ispin, spins );
            // forces_virtual[img][ispin] = Force_Single_Spin( ispin, gradient, spins, *this ).cross( spins[ispin] );

            // Heun_Propagator( 0.5, ispin, spins, *this );
            // std::cout << "update \n";

            // std::cout << "===== SECOND ====\n";
            // std::cout << "spins[ispin]" << spins[ispin].transpose() << "\n";

            // gradient_current         = this->System( 0 )->hamiltonian->Gradient_Single_Spin( ispin, spins );
            // gradient_linear_current  = this->System( 0 )->hamiltonian->Linear_Gradient_Contribution_Single_Spin(
            // ispin, spins ); spin_current             = spins[ispin];

            // auto rev_force_callback = [&solver = *this, gradient_current, gradient_linear_current, spin_current](
            // int ispin, const vectorfield & spins )
            // {
            //     // Vector3 gradient  = -solver.System( 0 )->hamiltonian->Gradient_Single_Spin( ispin, spins );
            //     Vector3 gradient  = -(gradient_current + gradient_linear_current * (spins[ispin] -
            //     spin_current)); Vector3 f         = Force_Single_Spin( ispin, gradient, spins, solver );
            //     // std::cout << "f " << f.transpose() << "\n";
            //     return -f;
            // };
            // Implicit_Propagator( 0.5, ispin, spins, rev_force_callback, *this );

            // std::cout << "spins[ispin]" << spins[ispin].transpose() << "\n";
            // std::cout << "gradient" << gradient_current.transpose() << "\n";

            // std::cout << "=====\n";
        }

        for( int ispin = nos - 1; ispin > -1; ispin-- )
        {
            // Compute this information so that we do not have to re-compute the entire gradient in every iteration
            gradient_current = this->System( 0 )->hamiltonian->Gradient_Single_Spin( ispin, spins );
            gradient_linear_current
                = this->System( 0 )->hamiltonian->Linear_Gradient_Contribution_Single_Spin( ispin, spins );
            spin_current = spins[ispin];

            // auto force_callback = [&solver = *this, gradient_current, gradient_linear_current,
            //                        spin_current]( int ispin, const vectorfield & spins ) {
            //     Vector3 gradient = -( gradient_current + gradient_linear_current * ( spins[ispin] - spin_current
            //     ) ); Vector3 f        = Force_Single_Spin( ispin, gradient, spins, solver ); return f;
            // };

            // auto gradient_callback = [&solver = *this, gradient_current, gradient_linear_current,
            //                           spin_current]( int ispin, const vectorfield & spins ) {
            //     Vector3 gradient = -( gradient_current + gradient_linear_current * ( spins[ispin] - spin_current
            //     ) ); return gradient;
            // };

            if( this->st_propagator == ST_Propagator::IMP )
            {
                Implicit_Propagator( 0.5, ispin, spins, force_callback, *this );
            }
            else
            {
                if( this->has_linear_self_contributions )
                {
                    SA_Implicit_Propagator( 0.5, ispin, spins, gradient_callback, *this );
                }
                else
                {
                    SA_Propagator( 0.5, -gradient_current, ispin, spins, *this );
                }
            }
            // Update convergence information
            Vector3 gradient           = force_callback( ispin, spins );
            forces_virtual[img][ispin] = Force_Single_Spin( ispin, gradient, spins, *this ).cross( spins[ispin] );
        }
    }
}

template<>
inline std::string Method_Solver<Solver::ST>::SolverName()
{
    if( this->st_propagator == ST_Propagator::IMP )
    {
        return "IMP_ST";
    }
    else
    {
        if( this->has_linear_self_contributions )
        {
            return "SA_ST (self. cons.)";
        }
        else
        {
            return "SA_ST";
        }
    }
}

template<>
inline std::string Method_Solver<Solver::ST>::SolverFullName()
{
    if( this->st_propagator == ST_Propagator::IMP )
    {
        return "Suzuki-Trotter (implicit)";
    }
    else
    {
        if( this->has_linear_self_contributions )
        {
            return "Suzuki-Trotter (spin aligned, self consistent)";
        }
        else
        {
            return "Suzuki-Trotter (spin aligned)";
        }
    }
}