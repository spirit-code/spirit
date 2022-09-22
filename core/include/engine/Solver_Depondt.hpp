#include "engine/Vectormath_Defines.hpp"
#include <Eigen/Geometry>
#include <engine/Backend_par.hpp>

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
        configurations_predictor[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, { 0, 0, 0 } ) );

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
        // clang-format off
        Backend::par::apply( nos, 
            [
                c = this->configurations[i]->data(),
                c_p = this->configurations_predictor[i]->data(),
                f = forces_virtual[i].data()
            ] SPIRIT_LAMBDA (int idx)
            {
                const Vector3 delta_s         = -c[idx].cross(f[idx]);
                const scalar angle            = delta_s.norm();
                const Vector3 axis            = c[idx].cross( delta_s );
                const Matrix3 rotation_matrix = Eigen::AngleAxis<scalar>(angle, axis.normalized()).toRotationMatrix();

                c_p[idx] = rotation_matrix*c[idx];
                c_p[idx].normalize();
            } 
        );
        // clang-format on
    }

    // Calculate_Force for the Corrector
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );

    // Predictor for each image
    for( int i = 0; i < this->noi; ++i )
    {
        // clang-format off
        Backend::par::apply( nos,
            [
                c = this->configurations[i]->data(),
                c_p = this->configurations_predictor[i]->data(),
                f = forces_virtual[i].data(),
                f_p = forces_virtual_predictor[i].data()
            ] SPIRIT_LAMBDA (int idx)
            {
                // Rotate predictor forces back into tangent frame of current spin
                f_p[idx] = f_p[idx] - f_p[idx].dot(c_p[idx]) * c_p[idx]; // Remove normal component of predictor

                // Average axis of rotation
                Vector3 avg_force = 0.5 * ( f[idx] + f_p[idx] );
                avg_force = avg_force - avg_force.dot( c[idx] ) * c[idx];

                const scalar angle      = avg_force.norm();
                const Vector3 axis      = avg_force.normalized();

                const Matrix3 rotation_matrix = Eigen::AngleAxis<scalar>(angle, axis.normalized()).toRotationMatrix();

                c[idx] = rotation_matrix*c[idx];
                c[idx].normalize();
            } 
        );
        // clang-format on
    }
}

template<>
inline std::string Method_Solver<Solver::Depondt>::SolverName()
{
    return "Depondt";
}

template<>
inline std::string Method_Solver<Solver::Depondt>::SolverFullName()
{
    return "Depondt";
}