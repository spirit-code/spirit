#include "data/Parameters_Method_LLG.hpp"
#include "data/Spin_System.hpp"
#include "engine/Vectormath_Defines.hpp"
#include <fmt/format.h>
#include <limits>

template<>
inline void Method_Solver<Solver::ST>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
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
    scalar time_scale, Vector3 & gradient, int ispin, vectorfield & spins, const Method_Solver<Solver::ST> & solver )
{
    const Vector3 ez = Vector3{ 0, 0, 1 };

    // Rotate into a frame where the gradient (not the force!) points into z direction
    const scalar angle = Angle( gradient, ez );
    const Vector3 axis = gradient.cross( ez );

    const Matrix3 rotation_matrix = Eigen::AngleAxis<scalar>( angle, axis.normalized() ).toRotationMatrix();

    Vector3 spin_temp = rotation_matrix * spins[ispin];
    spin_temp         = rotation_matrix.transpose() * analytical( time_scale, spin_temp, gradient.norm(), solver );

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

inline void
Implicit_Propagator( scalar time_scale, int ispin, vectorfield & spins, const Method_Solver<Solver::ST> & solver )
{
    // // Solve the equation s_{i+1} = s_{i} + 0.5 * (s_{i} + s_{i+1}) x ( B + 0.5 (A s_{i} + A s_{i+1}) )
    // bool run = true;

    // scalar convergence = 10 * std::numeric_limits<scalar>::epsilon();
    // int max_iter       = 100;

    // Vector3 spins_previous;
    // Matrix3 A;
    // Vectr3 B;

    // int iter = 0;

    // while( run )
    // {
    //     spin_previous = spins[ispin];
    //     spins[ispin] =

    //         sp = implicit_equation2( s, sp, B, Ak )

    //             change
    //         = np.linalg.norm( sp - s_prev ) converged = change < 1e-16

    //                                                     iter
    //         += 1 change_list.append( change )
    // }
}

template<>
inline void Method_Solver<Solver::ST>::Iteration()
{
    for( int img = 0; img < this->noi; ++img )
    {
        auto & spins = *this->systems[img]->spins;

        // Half time step in order
        for( int ispin = 0; ispin < this->nos; ispin++ )
        {
            Vector3 gradient           = -this->systems[img]->hamiltonian->Gradient_Single_Spin( ispin, spins );
            forces_virtual[img][ispin] = Force_Single_Spin( ispin, gradient, spins, *this ).cross( spins[ispin] );
            SA_Propagator( 0.5, gradient, ispin, spins, *this );
            // Heun_Propagator( 0.5, ispin, spins, *this );
        }

        // Half time step in reverse order
        for( int ispin = this->nos - 1; ispin >= 0; ispin-- )
        {
            Vector3 gradient           = -this->systems[img]->hamiltonian->Gradient_Single_Spin( ispin, spins );
            forces_virtual[img][ispin] = Force_Single_Spin( ispin, gradient, spins, *this ).cross( spins[ispin] );
            SA_Propagator( 0.5, gradient, ispin, spins, *this );
            // Heun_Propagator( 0.5, ispin, spins, *this );
        }
    }
}

template<>
inline std::string Method_Solver<Solver::ST>::SolverName()
{
    return "ST";
}

template<>
inline std::string Method_Solver<Solver::ST>::SolverFullName()
{
    return "Suzuki_Trotter";
}