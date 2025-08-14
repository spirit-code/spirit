#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP

#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Hamiltonian.hpp>
#include <engine/spin/Interaction_Wrapper.hpp>
#include <engine/spin/interaction/Anisotropy.hpp>
#include <engine/spin/interaction/Biaxial_Anisotropy.hpp>
#include <engine/spin/interaction/Cubic_Anisotropy.hpp>
#include <engine/spin/interaction/DDI.hpp>
#include <engine/spin/interaction/DMI.hpp>
#include <engine/spin/interaction/Exchange.hpp>
#include <engine/spin/interaction/Gaussian.hpp>
#include <engine/spin/interaction/Quadruplet.hpp>
#include <engine/spin/interaction/Two_Site_Anisotropy.hpp>
#include <engine/spin/interaction/Zeeman.hpp>
#include <utility/Variadic_Traits.hpp>

namespace Engine
{

namespace Spin
{

// clang-format off
using HamiltonianBase = Common::Hamiltonian<
    // interaction wrapping
    Interaction::IWrapper<StateType>,
    Interaction::Wrapper,
    // list of interactions
    Interaction::Zeeman,
    Interaction::Anisotropy,
    Interaction::Biaxial_Anisotropy,
    Interaction::Cubic_Anisotropy,
    Interaction::Exchange,
    Interaction::DMI,
    Interaction::Quadruplet,
    Interaction::DDI,
    Interaction::Two_Site_Anisotropy,
    Interaction::Gaussian>;
// clang-format on

// Hamiltonian for (pure) spin systems
struct Hamiltonian : public HamiltonianBase
{
public:
    using HamiltonianBase::Hamiltonian;

    using state_t = typename HamiltonianBase::state_t;

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        hessian.setZero();
        Backend::apply(
            [&state, hessian = Interaction::Functor::dense_hessian_wrapper( hessian )]( auto &... interaction )
            { ( ..., interaction.Hessian_Impl( state, hessian ) ); }, interactions );
    };

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        std::vector<Common::Interaction::triplet> tripletList;
        tripletList.reserve( get_geometry().n_cells_total * Sparse_Hessian_Size_per_Cell() );
        Backend::apply(
            [&state, hessian = Interaction::Functor::sparse_hessian_wrapper( tripletList )]( auto &... interaction )
            { ( ..., interaction.Hessian_Impl( state, hessian ) ); }, interactions );
        hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
    };

    std::size_t Sparse_Hessian_Size_per_Cell() const
    {
        return Backend::apply(
            []( const auto &... interaction ) -> std::size_t
            { return ( std::size_t( 0 ) + ... + interaction.Sparse_Hessian_Size_per_Cell() ); }, interactions );
    };

    [[nodiscard]] Vector3 Spin_Gradient_Local( const int ispin, const state_t & state )
    {
        return Backend::apply(
            [ispin, &state]( auto &... interaction ) -> Vector3
            { return ( Vector3::Zero() + ... + interaction.Spin_Gradient_Local( ispin, state ) ); }, interactions );
    };

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        const auto nos = state.spin.size();

        if( gradient.size() != nos )
            gradient = vectorfield( nos, Vector3::Zero() );
        else
            Vectormath::fill( gradient, Vector3::Zero() );

        Backend::apply(
            [&state, &gradient]( auto &... interaction ) { ( ..., interaction.Gradient( state, gradient ) ); },
            interactions );
    };

    // provided for backwards compatibility, this function no longer serves a purpose
    [[nodiscard]] scalar Gradient_and_Energy( const state_t & state, vectorfield & gradient )
    {
        Gradient( state, gradient );
        return Energy( state );
    };

    [[nodiscard]] std::string_view Name() const noexcept
    {
        if( !this->template is_contributing<Interaction::Gaussian>() )
            return "Heisenberg";

        auto gaussian_func = []( auto &... interaction )
        {
            return (
                true && ...
                && ( std::is_same_v<
                         typename std::decay_t<decltype( interaction )>::Interaction, Spin::Interaction::Gaussian>
                     || !interaction.is_contributing() ) );
        };

        if( Backend::apply( gaussian_func, interactions ) )
            return "Gaussian";

        return "Unknown";
    };
};

} // namespace Spin

} // namespace Engine

#endif
