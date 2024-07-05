#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_SOLVER_KERNELS_HPP
#define SPIRIT_CORE_ENGINE_SPIN_SOLVER_KERNELS_HPP

#include <engine/Vectormath_Defines.hpp>

#include <vector>

namespace Engine
{

namespace Solver_Kernels
{

namespace VP
{

void bare_velocity( const vectorfield & force, const vectorfield & force_previous, vectorfield & velocity );

void projected_velocity( const Vector2 projection, const vectorfield & force, vectorfield & velocity );

template<bool is_torque = true>
void apply_velocity(
    const vectorfield & velocity, const vectorfield & force, const scalar dt, vectorfield & configuration );

void set_step( const vectorfield & velocity, const vectorfield & force, const scalar dt, vectorfield & out );

}; // namespace VP

// SIB
void sib_transform( const vectorfield & spins, const vectorfield & force, vectorfield & out );

// Heun
template<bool is_torque = true>
void heun_predictor(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & delta_configurations,
    vectorfield & configurations_predictor );

template<bool is_torque = true>
void heun_corrector(
    const vectorfield & force_virtual_predictor, const vectorfield & delta_configuration,
    const vectorfield & configuration_predictor, vectorfield & configuration );

// RungeKutta4
template<bool is_torque = true>
void rk4_predictor_1(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & delta_configuration,
    vectorfield & configurations_predictor );

template<bool is_torque = true>
void rk4_predictor_2(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & delta_configuration,
    vectorfield & configurations_predictor );

template<bool is_torque = true>
void rk4_predictor_3(
    const vectorfield & configuration, const vectorfield & forces_virtual, vectorfield & delta_configuration,
    vectorfield & configurations_predictor );

template<bool is_torque = true>
void rk4_corrector(
    const vectorfield & forces_virtual, const vectorfield & configurations_k1, const vectorfield & configurations_k2,
    const vectorfield & configurations_k3, const vectorfield & configurations_predictor, vectorfield & configurations );

// Depondt
void depondt_predictor(
    const vectorfield & force_virtual, vectorfield & axis, scalarfield & angle, const vectorfield & configuration,
    vectorfield & predictor );

void depondt_corrector(
    const vectorfield & force_virtual, const vectorfield & force_virtual_predictor, vectorfield & axis,
    scalarfield & angle, vectorfield & configuration );

// OSO coordinates
void oso_rotate( vectorfield & spins, const vectorfield & searchdir );
void oso_calc_gradients( vectorfield & residuals, const vectorfield & spins, const vectorfield & forces );
scalar maximum_rotation( const vectorfield & searchdir, scalar maxmove );

// Atlas coordinates
void atlas_calc_gradients(
    vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords );
void atlas_rotate( vectorfield & spins, const scalarfield & a3_coords, const vector2field & searchdir );
bool ncg_atlas_check_coordinates( const vectorfield & spins, const scalarfield & a3_coords, scalar tol = -0.6 );
void lbfgs_atlas_transform_direction(
    const vectorfield & spins, scalarfield & a3_coords, std::vector<vector2field> & atlas_updates,
    std::vector<vector2field> & grad_updates, vector2field & searchdir, vector2field & grad_pr, scalarfield & rho_inv );

// LBFGS
template<typename Vec>
void lbfgs_get_searchdir(
    int & local_iter, scalarfield & rho, scalarfield & alpha, std::vector<field<Vec>> & q_vec,
    std::vector<field<Vec>> & searchdir, std::vector<std::vector<field<Vec>>> & delta_a,
    std::vector<std::vector<field<Vec>>> & delta_grad, const std::vector<field<Vec>> & grad,
    std::vector<field<Vec>> & grad_pr, const int num_mem, const scalar maxmove );

} // namespace Solver_Kernels

} // namespace Engine

#include <engine/Solver_Kernels.inl>

#endif
