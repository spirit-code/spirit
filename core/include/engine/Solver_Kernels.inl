#pragma once

#include <engine/Backend.hpp>
#include <engine/Solver_Kernels.hpp>
#include <engine/Vectormath.hpp>

#include <Eigen/Core>

namespace Engine
{

namespace Solver_Kernels
{

namespace VP
{

static constexpr scalar mass = 1.0;

inline void bare_velocity( const vectorfield & force, const vectorfield & force_previous, vectorfield & velocity )
{
    const auto * f    = force.data();
    const auto * f_pr = force_previous.data();
    auto * v          = velocity.data();

    // Calculate the new velocity
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), velocity.size(),
        [f, f_pr, v] SPIRIT_LAMBDA( const int idx ) { v[idx] += 0.5 / mass * ( f_pr[idx] + f[idx] ); } );
}

inline void projected_velocity( const Vector2 projection, const vectorfield & force, vectorfield & velocity )
{
    const auto * f = force.data();
    auto * v       = velocity.data();

    // Calculate the projected velocity
    if( projection[0] <= 0 )
    {
        Vectormath::fill( velocity, { 0, 0, 0 } );
    }
    else
    {
        const scalar ratio = projection[0] / projection[1];
        Backend::for_each_n(
            SPIRIT_PAR Backend::make_counting_iterator( 0 ), force.size(),
            [f, v, ratio] SPIRIT_LAMBDA( const int idx ) { v[idx] = f[idx] * ratio; } );
    }
}

template<bool is_torque>
void apply_velocity(
    const vectorfield & velocity, const vectorfield & force, const scalar dt, vectorfield & configuration )
{
    const auto * f = force.data();
    const auto * v = velocity.data();
    auto * conf    = configuration.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), force.size(),
        [conf, dt, v, f] SPIRIT_LAMBDA( const int idx )
        {
            conf[idx] += dt * ( v[idx] + 0.5 / mass * f[idx] );
            if constexpr( is_torque )
                conf[idx].normalize();
        } );
}

inline void set_step( const vectorfield & velocity, const vectorfield & force, const scalar dt, vectorfield & out )
{
    const auto * f = force.data();
    const auto * v = velocity.data();
    auto * o       = out.data();

    Backend::for_each_n(
        SPIRIT_PAR Backend::make_counting_iterator( 0 ), force.size(),
        [o, dt, v, f] SPIRIT_LAMBDA( const int idx ) { o[idx] = dt * ( v[idx] + 0.5 / mass * f[idx] ); } );
}

}; // namespace VP

// SIB
void sib_transform( const vectorfield & spins, const vectorfield & force, vectorfield & out );

// Heun
// torque version of the Heun predictor (assumes normalized configuration vectors)
template<>
inline void heun_predictor<true>(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & delta_configuration,
    vectorfield & configurations_predictor )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            configuration.begin(), forces_virtual.begin(), delta_configuration.begin(),
            configurations_predictor.begin() ),
        configuration.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( const Vector3 & conf, const Vector3 & force, Vector3 & delta, Vector3 & predictor )
            {
                delta     = -conf.cross( force );
                predictor = ( conf + delta ).normalized();
            } ) );
}

// torque version of the Heun corrector (assumes normalized configuration vectors)
template<>
inline void heun_corrector<true>(
    const vectorfield & force_virtual_predictor, const vectorfield & delta_configuration,
    const vectorfield & configuration_predictor, vectorfield & configuration )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            configuration.begin(), force_virtual_predictor.begin(), delta_configuration.begin(),
            configuration_predictor.begin() ),
        configuration.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( Vector3 & conf, const Vector3 & torque, const Vector3 & delta, const Vector3 & conf_pred )
            {
                // conf = conf + 0.5 * configurations_temp - 0.5 * ( conf' x A' )
                conf += 0.5 * ( delta - conf_pred.cross( torque ) );
                conf.normalize();
            } ) );
}

// RungeKutta4
// torque version of the 4th order Runge-Kutta predictor (assumes normalized configuration vectors)
template<>
inline void rk4_predictor_1<true>(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & k1,
    vectorfield & configurations_predictor )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            configuration.begin(), forces_virtual.begin(), k1.begin(), configurations_predictor.begin() ),
        configuration.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( const Vector3 & conf, const Vector3 & force, Vector3 & delta, Vector3 & conf_pred )
            {
                delta     = -conf.cross( force );
                conf_pred = ( conf + 0.5 * delta ).normalized();
            } ) );
}

template<>
inline void rk4_predictor_2<true>(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & k2,
    vectorfield & configurations_predictor )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            configuration.begin(), forces_virtual.begin(), k2.begin(), configurations_predictor.begin() ),
        configuration.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( const Vector3 & conf, const Vector3 & force, Vector3 & delta, Vector3 & conf_pred )
            {
                delta     = -conf_pred.cross( force );
                conf_pred = ( conf + 0.5 * delta ).normalized();
            } ) );
}

template<>
inline void rk4_predictor_3<true>(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & k3,
    vectorfield & configurations_predictor )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            configuration.begin(), forces_virtual.begin(), k3.begin(), configurations_predictor.begin() ),
        configuration.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( const Vector3 & conf, const Vector3 & force, Vector3 & delta, Vector3 & conf_pred )
            {
                delta     = -conf_pred.cross( force );
                conf_pred = ( conf + delta ).normalized();
            } ) );
}

// torque version of the 4th order Runge-Kutta corrector (assumes normalized configuration vectors)
template<>
inline void rk4_corrector<true>(
    const vectorfield & forces_virtual, const vectorfield & configurations_k1, const vectorfield & configurations_k2,
    const vectorfield & configurations_k3, const vectorfield & configurations_predictor, vectorfield & configurations )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            forces_virtual.begin(), configurations_k1.begin(), configurations_k2.begin(), configurations_k3.begin(),
            configurations_predictor.begin(), configurations.begin() ),
        configurations.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA(
                const Vector3 & torque, const Vector3 & k1, const Vector3 & k2, const Vector3 & k3,
                const Vector3 & conf_pred, Vector3 & conf )
            {
                conf
                    += 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 - 1.0 / 6.0 * /*-k4=*/conf_pred.cross( torque );
                conf.normalize();
            } ) );
}

// Depondt
inline void depondt_predictor(
    const vectorfield & force_virtual, vectorfield & axis, scalarfield & angle, const vectorfield & configuration,
    vectorfield & predictor )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator( force_virtual.begin(), axis.begin(), angle.begin() ),
        force_virtual.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( const Vector3 & force, Vector3 & axis, scalar & angle )
            {
                // For Rotation matrix R := R( H_normed, angle )
                angle = force.norm();
                // Normalize axis to get rotation axes
                axis = force.normalized();
            } ) );

    // Get spin predictor n' = R(H) * n
    Vectormath::rotate( configuration, axis, angle, predictor );
};

inline void depondt_corrector(
    const vectorfield & force_virtual, const vectorfield & force_virtual_predictor, vectorfield & axis,
    scalarfield & angle, vectorfield & configuration )
{
    Backend::for_each_n(
        SPIRIT_PAR Backend::make_zip_iterator(
            force_virtual.begin(), force_virtual_predictor.begin(), axis.begin(), angle.begin() ),
        force_virtual.size(),
        Backend::make_zip_function(
            [] SPIRIT_LAMBDA( const Vector3 & force, const Vector3 & force_predictor, Vector3 & axis, scalar & angle )
            {
                // Calculate the linear combination of the two forces_virtuals
                axis = 0.5 * ( force + force_predictor ); // H = (H + H')/2
                // Get the rotation angle as norm of temp1 ...For Rotation matrix R' := R( H'_normed, angle' )
                angle = axis.norm();
                // Normalize axis to get rotation axes
                axis.normalize();
            } ) );

    // Get new spin conf n_new = R( (H+H')/2 ) * n
    Vectormath::rotate( configuration, axis, angle, configuration );
};

// LBFGS
template<typename Vec>
void lbfgs_get_searchdir(
    int & local_iter, scalarfield & rho, scalarfield & alpha, std::vector<field<Vec>> & q_vec,
    std::vector<field<Vec>> & searchdir, std::vector<std::vector<field<Vec>>> & delta_a,
    std::vector<std::vector<field<Vec>>> & delta_grad, const std::vector<field<Vec>> & grad,
    std::vector<field<Vec>> & grad_pr, const int num_mem, const scalar maxmove )
{
    using std::begin, std::end;

    // std::cerr << "lbfgs searchdir \n";
    static auto dot     = [] SPIRIT_LAMBDA( const Vec & v1, const Vec & v2 ) { return v1.dot( v2 ); };
    static auto set     = [] SPIRIT_LAMBDA( const Vec & x ) { return x; };
    static auto inverse = [] SPIRIT_LAMBDA( const Vec & x ) { return -x; };

    static constexpr scalar epsilon = std::is_same_v<scalar, float> ? 1e-30 : 1e-300;

    const int noi     = grad.size();
    const int m_index = local_iter % num_mem; // memory index
    int c_ind         = 0;

    if( local_iter == 0 ) // gradient descent
    {
        for( int img = 0; img < noi; img++ )
        {
            Backend::copy( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( grad_pr[img] ) );
            Backend::transform( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( searchdir[img] ), inverse );

            auto & da = delta_a[img];
            auto & dg = delta_grad[img];
            Backend::fill_n( begin( rho ), num_mem, 0.0 );
            for( int i = 0; i < num_mem; i++ )
            {
                Backend::fill( begin( delta_a[img][i] ), end( delta_a[img][i] ), Vec::Zero() );
                Backend::fill( begin( delta_grad[img][i] ), end( delta_grad[img][i] ), Vec::Zero() );
            }
        }
    }
    else
    {
        for( int img = 0; img < noi; img++ )
        {
            Backend::copy( SPIRIT_PAR begin( searchdir[img] ), end( searchdir[img] ), begin( delta_a[img][m_index] ) );
            Backend::transform(
                SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( grad_pr[img] ),
                begin( delta_grad[img][m_index] ),
                [] SPIRIT_LAMBDA( const Vec & g, const Vec & g_pr ) { return g - g_pr; } );
        }

        scalar rinv_temp = 0;
        for( int img = 0; img < noi; img++ )
            rinv_temp += Backend::transform_reduce(
                SPIRIT_PAR begin( delta_grad[img][m_index] ), end( delta_grad[img][m_index] ),
                begin( delta_a[img][m_index] ), scalar( 0 ), Backend::plus<scalar>{}, dot );

        if( rinv_temp > epsilon )
            rho[m_index] = 1.0 / rinv_temp;
        else
        {
            local_iter = 0;
            return lbfgs_get_searchdir(
                local_iter, rho, alpha, q_vec, searchdir, delta_a, delta_grad, grad, grad_pr, num_mem, maxmove );
        }

        for( int img = 0; img < noi; img++ )
            Backend::copy( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( q_vec[img] ) );

        for( int k = num_mem - 1; k > -1; k-- )
        {
            c_ind       = ( k + m_index + 1 ) % num_mem;
            scalar temp = 0;
            for( int img = 0; img < noi; img++ )
                temp += Backend::transform_reduce(
                    SPIRIT_PAR begin( delta_a[img][c_ind] ), end( delta_a[img][c_ind] ), begin( q_vec[img] ),
                    scalar( 0 ), Backend::plus<scalar>{}, dot );

            alpha[c_ind] = rho[c_ind] * temp;
            for( int img = 0; img < noi; img++ )
            {
                auto a = alpha[c_ind];
                Backend::transform(
                    SPIRIT_PAR begin( q_vec[img] ), end( q_vec[img] ), begin( delta_grad[img][c_ind] ),
                    begin( q_vec[img] ),
                    [a] SPIRIT_LAMBDA( const Vec & q, const Vec & d ) -> Vec { return q - a * d; } );
            }
        }

        scalar dy2 = 0;
        for( int img = 0; img < noi; img++ )
            dy2 += Backend::transform_reduce(
                SPIRIT_PAR begin( delta_grad[img][m_index] ), end( delta_grad[img][m_index] ),
                begin( delta_grad[img][m_index] ), scalar( 0 ), Backend::plus<scalar>{}, dot );

        for( int img = 0; img < noi; img++ )
        {
            scalar rhody2     = dy2 * rho[m_index];
            scalar inv_rhody2 = 0.0;
            if( rhody2 > epsilon )
                inv_rhody2 = 1.0 / rhody2;
            else
                inv_rhody2 = 1.0 / ( epsilon );
            Backend::transform(
                SPIRIT_PAR begin( q_vec[img] ), end( q_vec[img] ), begin( searchdir[img] ),
                [inv_rhody2] SPIRIT_LAMBDA( const Vec & q ) { return inv_rhody2 * q; } );
        }

        for( int k = 0; k < num_mem; k++ )
        {
            if( local_iter < num_mem )
                c_ind = k;
            else
                c_ind = ( k + m_index + 1 ) % num_mem;

            scalar rhopdg = 0;
            for( int img = 0; img < noi; img++ )
                rhopdg += Backend::transform_reduce(
                    SPIRIT_PAR begin( delta_grad[img][c_ind] ), end( delta_grad[img][c_ind] ), begin( searchdir[img] ),
                    scalar( 0 ), Backend::plus<scalar>{}, dot );

            rhopdg *= rho[c_ind];

            for( int img = 0; img < noi; img++ )
            {
                const auto alph = alpha[c_ind] - rhopdg;
                Backend::transform(
                    SPIRIT_PAR begin( searchdir[img] ), end( searchdir[img] ), begin( delta_a[img][c_ind] ),
                    begin( searchdir[img] ),
                    [alph] SPIRIT_LAMBDA( const Vec & sd, const Vec & da ) { return sd + alph * da; } );
            }
        }

        for( int img = 0; img < noi; img++ )
        {
            Backend::transform(
                SPIRIT_PAR begin( searchdir[img] ), end( searchdir[img] ), begin( searchdir[img] ), inverse );
            Backend::copy( SPIRIT_PAR begin( grad[img] ), end( grad[img] ), begin( grad_pr[img] ) );
        }
    }
    local_iter++;
}

} // namespace Solver_Kernels

} // namespace Engine
