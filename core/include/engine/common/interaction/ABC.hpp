#pragma once

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <memory>
#include <optional>

namespace Engine
{

namespace Common
{

namespace Interaction
{

using triplet = Eigen::Triplet<scalar>;

class ABC;

/*
 * Interface over which an Interaction has access to (part) of the Hamiltonian that owns it.
 * This also serves as an Observer interface where an interaction may notify the Hamiltonian of changes to it through
 * the `onInteractionChanged()` method. Additionally, all interactions share the same Geometry and boundary conditions.
 * So, these data members must be made accessible through this interface as well.
 */
struct Owner
{
    friend void swap( Owner & first, Owner & second ) noexcept
    {
        using std::swap;
        if( &first == &second )
        {
            return;
        }

        swap( first.geometry, second.geometry );
        swap( first.boundary_conditions, second.boundary_conditions );
    };

    virtual ~Owner() = default;

    /*
     * virtual member function to trigger updates in the collection of interactions that the owner has when one
     * (or more) of the interactions has been changed
     */
    void virtual onInteractionChanged()        = 0;
    void virtual onGeometryChanged()           = 0;
    void virtual onBoundaryConditionsChanged() = 0;

    // setters with correspinding update functions
    void setGeometry( std::shared_ptr<Data::Geometry> g )
    {
        geometry = std::move( g );
        onGeometryChanged();
    }

    void setBoundaryConditions( const intfield & bc )
    {
        boundary_conditions = bc;
        onBoundaryConditionsChanged();
    }

    // getters provide constant references to ensure data integrety
    const Data::Geometry & getGeometry() const noexcept
    {
        return *geometry;
    }

    const intfield & getBoundaryConditions() const noexcept
    {
        return boundary_conditions;
    }

protected:
    std::shared_ptr<Data::Geometry> geometry;
    intfield boundary_conditions;

    Owner( std::shared_ptr<Data::Geometry> && geometry, intfield && boundary_conditions ) noexcept
            : geometry( geometry ), boundary_conditions( boundary_conditions ){};

    Owner( const std::shared_ptr<Data::Geometry> & geometry, const intfield & boundary_conditions )
            : geometry( geometry ), boundary_conditions( boundary_conditions ){};
};

void setOwnerPtr( ABC & interaction, Owner * owner ) noexcept;

/*
 * Abstract base class that specifies the interface any interaction must have.
 */
class ABC
{
    friend void setOwnerPtr( ABC & interaction, Owner * owner ) noexcept;

public:
    virtual ~ABC() = default;

    // Interaction name as string (must be unique per interaction because interactions with the same name cannot exist
    // within the same hamiltonian at the same time)
    [[nodiscard]] virtual std::string_view Name() const = 0;

    [[nodiscard]] virtual bool is_contributing() const
    {
        return true;
    };

    [[nodiscard]] virtual bool is_enabled() const final
    {
        return enabled;
    };

    [[nodiscard]] virtual bool is_active() const final
    {
        return is_enabled() && is_contributing();
    };

    virtual void enable() final
    {
        this->enabled = true;
        hamiltonian->onInteractionChanged();
    };

    virtual void disable() final
    {
        this->enabled = false;
        hamiltonian->onInteractionChanged();
    };

    virtual void updateGeometry() final
    {
        this->updateFromGeometry( this->hamiltonian->getGeometry() );
    };

protected:
    ABC( Owner * hamiltonian, scalarfield energy_per_spin, scalar delta = 1e-3 ) noexcept
            : energy_per_spin( std::move( energy_per_spin ) ), delta( delta ), hamiltonian( hamiltonian ){};

    virtual void updateFromGeometry( const Data::Geometry & geometry ) = 0;

    // local compute buffer
    scalarfield energy_per_spin;

    scalar delta = 1e-3;

    // maybe used for the GUI
    bool enabled = true;

#if defined( SPIRIT_USE_OPENMP ) || defined( SPIRIT_USE_CUDA )
    // When parallelising (cuda or openmp), we need all neighbours per spin
    static constexpr bool use_redundant_neighbours = true;
#else
    // When running on a single thread, we can ignore redundant neighbours
    static constexpr bool use_redundant_neighbours = false;
#endif

    // as long as the interaction is only constructible inside the Hamiltonian,
    // it is safe to assume that the Hamiltonian pointed to always exists
    Owner * hamiltonian;

private:
    static constexpr std::string_view name = "Common::Interaction::ABC";
};

/*
 * Intermediary Interaction Base class to inherit from (using CRTP) when implementing an Interaction
 * This class inherits from the abstract base class defining the interface for the specific interaction type
 * (`{namespace}::Interaction::ABC`). It then implements the common functionality for all Interaction types with
 * specific knowledge of the specific implementation of the Interaction class. The full hirarchy of `Derived` reads:
 *      Common::Interaction::ABC
 *          -> {namespace}::Interaction::ABC
 *          -> Common::Interaction::Base<{namespace}::Interaction::ABC, Derived>
 *          -> {namespace}::Interaction::Base<Derived>
 *          -> Derived
 *  where `{namespace}::Interaction::ABC` is an intermediary abstract base class defining the interface of
 */
template<class ABC, class Derived>
class Base : public ABC
{
protected:
    Base( Owner * hamiltonian, scalarfield energy_per_spin, scalar delta ) noexcept
            : ABC( hamiltonian, energy_per_spin, delta ){};

    const Data::Geometry & getGeometry()
    {
        return this->hamiltonian->getGeometry();
    }

    const intfield & getBoundaryConditions()
    {
        return this->hamiltonian->getBoundaryConditions();
    }

public:
    [[nodiscard]] std::unique_ptr<ABC> clone( Common::Interaction::Owner * const owner ) const final
    {
        auto copy         = std::make_unique<Derived>( static_cast<const Derived &>( *this ) );
        copy->hamiltonian = owner;
        return copy;
    }

    [[nodiscard]] std::string_view Name() const final
    {
        return Derived::name;
    }
};

inline void setOwnerPtr( ABC & interaction, Owner * const owner ) noexcept
{
    interaction.hamiltonian = owner;
};

} // namespace Interaction

} // namespace Common

} // namespace Engine
