#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <data/Misc.hpp>
#include <engine/FFT.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Hamiltonian.hpp>
#include <engine/spin/Hamiltonian_Defines.hpp>
#include <engine/spin/interaction/ABC.hpp>
#include <engine/spin/interaction/Anisotropy.hpp>
#include <engine/spin/interaction/Biaxial_Anisotropy.hpp>
#include <engine/spin/interaction/Cubic_Anisotropy.hpp>
#include <engine/spin/interaction/DDI.hpp>
#include <engine/spin/interaction/DMI.hpp>
#include <engine/spin/interaction/Exchange.hpp>
#include <engine/spin/interaction/Gaussian.hpp>
#include <engine/spin/interaction/Quadruplet.hpp>
#include <engine/spin/interaction/Zeeman.hpp>
#include <utility/Span.hpp>

#include <memory>
#include <vector>

namespace Engine
{

namespace Spin
{

enum class HAMILTONIAN_CLASS
{
    GENERIC    = SPIRIT_HAMILTONIAN_CLASS_GENERIC,
    GAUSSIAN   = SPIRIT_HAMILTONIAN_CLASS_GAUSSIAN,
    HEISENBERG = SPIRIT_HAMILTONIAN_CLASS_HEISENBERG,
};

static constexpr std::string_view hamiltonianClassName( HAMILTONIAN_CLASS cls )
{
    switch( cls )
    {
        case HAMILTONIAN_CLASS::GENERIC: return "Generic";
        case HAMILTONIAN_CLASS::GAUSSIAN: return "Gaussian";
        case HAMILTONIAN_CLASS::HEISENBERG: return "Heisenberg";
        default: return "Unknown";
    };
}

/*
    The Heisenberg Hamiltonian for (pure) spin systems
*/
class Hamiltonian : public Common::Hamiltonian<Spin::Interaction::ABC>
{
public:
    using interaction_t = Spin::Interaction::ABC;
    using state_t       = typename interaction_t::state_t;

    Hamiltonian( std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions )
            : Common::Hamiltonian<interaction_t>( std::move( geometry ), std::move( boundary_conditions ) ),
              hamiltonian_class( HAMILTONIAN_CLASS::GENERIC )
    {
        this->updateName();
    };

    void Hessian( const vectorfield & spins, MatrixX & hessian );
    void Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian );

    void Gradient( const vectorfield & spins, vectorfield & gradient );
    void Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy );

    void Energy_per_Spin( const vectorfield & spins, scalarfield & contributions );
    void Energy_Contributions_per_Spin( const vectorfield & spins, Data::vectorlabeled<scalarfield> & contributions );
    Data::vectorlabeled<scalar> Energy_Contributions( const vectorfield & spins );

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    [[nodiscard]] scalar Energy_Single_Spin( int ispin, const vectorfield & spins );

    [[nodiscard]] scalar Energy( const vectorfield & spins );

    void Gradient_FD( const vectorfield & spins, vectorfield & gradient );
    void Hessian_FD( const vectorfield & spins, MatrixX & hessian );

    void updateName_Impl() final;
    [[nodiscard]] std::string_view Name() const final
    {
        return class_name;
    };

private:
    void partitionActiveInteractions( value_iterator first, value_iterator last ) final
    {
        // sort by spin order (may speed up predictions)
        const auto has_common_spin_order     = []( const auto & i ) { return i->spin_order() == common_spin_order; };
        const auto common_partition_boundary = std::partition( first, last, has_common_spin_order );
        common_interactions_size             = std::distance( begin( interactions ), common_partition_boundary );
    }
    [[nodiscard]] constexpr auto getCommonInteractions() const -> Utility::Span<const value_t>
    {
        return Utility::Span( begin( getActiveInteractions() ), common_interactions_size );
    };

    [[nodiscard]] constexpr auto getUncommonInteractions() const -> Utility::Span<const value_t>
    {
        const auto active_ = getActiveInteractions();
        return Utility::Span( begin( active_ ) + common_interactions_size, end( active_ ) );
    };

    scalar delta = 1e-3;

    // naming mechanism for compatibility with the named subclasses architecture
    HAMILTONIAN_CLASS hamiltonian_class = HAMILTONIAN_CLASS::GENERIC;
    std::string_view class_name{ hamiltonianClassName( HAMILTONIAN_CLASS::GENERIC ) };

    std::size_t common_interactions_size = 0;
    static constexpr int common_spin_order = 2;
};

} // namespace Spin

} // namespace Engine

#endif
