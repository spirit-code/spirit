#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP

#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Hamiltonian.hpp>
#include <engine/spin/Interaction_Standalone_Adaptor.hpp>
#include <engine/spin/interaction/Anisotropy.hpp>
#include <engine/spin/interaction/Biaxial_Anisotropy.hpp>
#include <engine/spin/interaction/Cubic_Anisotropy.hpp>
#include <engine/spin/interaction/DDI.hpp>
#include <engine/spin/interaction/DMI.hpp>
#include <engine/spin/interaction/Exchange.hpp>
#include <engine/spin/interaction/Gaussian.hpp>
#include <engine/spin/interaction/Quadruplet.hpp>
#include <engine/spin/interaction/Zeeman.hpp>
#include <utility/Variadic_Traits.hpp>

namespace Engine
{

namespace Spin
{
// TODO: look into mixins and decide if they are more suitable to compose the `Hamiltonian` and `StandaloneAdaptor` types

namespace Accessor = Common::Accessor;
namespace Functor  = Common::Functor;

// Hamiltonian for (pure) spin systems
template<typename state_type, typename StandaloneAdaptorType, typename... InteractionTypes>
class Hamiltonian : public Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>
{
    using base_t = Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>;

public:
    using Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>::Hamiltonian;

    using state_t = state_type;

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        hessian.setZero();
        Hessian_Impl( state, Interaction::Functor::dense_hessian_wrapper( hessian ) );
    };

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        std::vector<Common::Interaction::triplet> tripletList;
        tripletList.reserve( this->get_geometry().n_cells_total * Sparse_Hessian_Size_per_Cell() );
        Hessian_Impl( state, Interaction::Functor::sparse_hessian_wrapper( tripletList ) );
        hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
    };

    std::size_t Sparse_Hessian_Size_per_Cell() const
    {
        auto func = []( const auto &... interaction ) -> std::size_t
        { return ( std::size_t( 0 ) + ... + interaction.Sparse_Hessian_Size_per_Cell() ); };
        return Backend::apply( func, this->local ) + Backend::apply( func, this->nonlocal );
    };

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        const auto nos = state.size();

        if( gradient.size() != nos )
            gradient = vectorfield( nos, Vector3::Zero() );
        else
            Vectormath::fill( gradient, Vector3::Zero() );

        Backend::transform(
            SPIRIT_PAR this->indices.begin(), this->indices.end(), gradient.begin(),
            Functor::transform_op(
                Functor::tuple_dispatch<Accessor::Gradient>( this->local ), Vector3{ 0.0, 0.0, 0.0 }, state ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Gradient>( this->nonlocal ), state, gradient );
    };

    // provided for backwards compatibility, this function no longer serves a purpose
    [[nodiscard]] scalar Gradient_and_Energy( const state_t & state, vectorfield & gradient )
    {
        Gradient( state, gradient );
        return this->Energy( state );
    };

private:
    template<typename Callable>
    void Hessian_Impl( const state_t & state, Callable hessian )
    {
        Backend::cpu::for_each(
            this->indices.begin(), this->indices.end(),
            Functor::for_each_op( Functor::tuple_dispatch<Accessor::Hessian>( this->local ), state, hessian ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Hessian>( this->nonlocal ), state, hessian );
    };
};

struct HamiltonianVariantTypes
{
    using state_t     = vectorfield;
    using AdaptorType = Spin::Interaction::StandaloneAdaptor<state_t>;

    using Gaussian   = Hamiltonian<state_t, AdaptorType, Interaction::Gaussian>;
    using Heisenberg = Hamiltonian<
        state_t, AdaptorType, Interaction::Zeeman, Interaction::Anisotropy, Interaction::Biaxial_Anisotropy,
        Interaction::Cubic_Anisotropy, Interaction::Exchange, Interaction::DMI, Interaction::Quadruplet,
        Interaction::DDI>;

    using Variant = std::variant<Gaussian, Heisenberg>;
};

// Single Type wrapper around Variant Hamiltonian type
// Should the visitors split up into standalone function objects?
class HamiltonianVariant : public Common::HamiltonianVariant<HamiltonianVariant, HamiltonianVariantTypes>
{
public:
    using state_t     = typename HamiltonianVariantTypes::state_t;
    using Gaussian    = typename HamiltonianVariantTypes::Gaussian;
    using Heisenberg  = typename HamiltonianVariantTypes::Heisenberg;
    using Variant     = typename HamiltonianVariantTypes::Variant;
    using AdaptorType = typename HamiltonianVariantTypes::AdaptorType;

private:
    using base_t = Common::HamiltonianVariant<HamiltonianVariant, HamiltonianVariantTypes>;

public:
    explicit HamiltonianVariant( Gaussian && gaussian ) noexcept( std::is_nothrow_move_constructible_v<Gaussian> )
            : base_t( std::move( gaussian ) ) {};

    explicit HamiltonianVariant( Heisenberg && heisenberg ) noexcept( std::is_nothrow_move_constructible_v<Heisenberg> )
            : base_t( std::move( heisenberg ) ) {};

    [[nodiscard]] std::string_view Name() const noexcept
    {
        if( std::holds_alternative<Gaussian>( hamiltonian ) )
            return "Gaussian";

        if( std::holds_alternative<Heisenberg>( hamiltonian ) )
            return "Heisenberg";

        // std::unreachable();

        return "Unknown";
    };

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        std::visit( [&state, &gradient]( auto & h ) { h.Gradient( state, gradient ); }, hamiltonian );
    }

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        std::visit( [&state, &hessian]( auto & h ) { h.Hessian( state, hessian ); }, hamiltonian );
    }

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        std::visit( [&state, &hessian]( auto & h ) { h.Sparse_Hessian( state, hessian ); }, hamiltonian );
    }

    void Gradient_and_Energy( const state_t & state, vectorfield & gradient, scalar & energy )
    {
        std::visit(
            [&state, &gradient, &energy]( auto & h ) { energy = h.Gradient_and_Energy( state, gradient ); },
            hamiltonian );
    };
};

} // namespace Spin

} // namespace Engine

#endif
