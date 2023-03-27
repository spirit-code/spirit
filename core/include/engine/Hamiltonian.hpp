#pragma once
#ifndef SPIRIT_CORE_ENGINE_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_HAMILTONIAN_HPP

#include <Spirit/Spirit_Defines.h>
#include <engine/Vectormath_Defines.hpp>

#include <vector>

namespace Engine
{

/*
 * The Hamiltonian contains the interaction parameters of a System.
 * It also defines the functions to calculate the Effective Field and Energy.
 */
class Hamiltonian
{
public:
    Hamiltonian( intfield boundary_conditions );

    virtual ~Hamiltonian() = default;

    /*
     * Update the Energy array.
     * This needs to be done every time the parameters are changed, in case an energy
     * contribution is now non-zero or vice versa.
     */
    virtual void Update_Energy_Contributions() = 0;

    /*
     * Calculate the Hessian matrix of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient. You should
     * override it if you want to get proper performance.
     * This function is the fallback for derived classes where it has not been overridden.
     */
    virtual void Hessian( const vectorfield & spins, MatrixX & hessian ) = 0;

    /*
     */
    virtual void Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian ) = 0;

    /*
     * Calculate the energy gradient of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient. You should
     * override it if you want to get proper performance.
     * This function is the fallback for derived classes where it has not been overridden.
     */
    virtual void Gradient( const vectorfield & spins, vectorfield & gradient ) = 0;

    /*
     * Calculates the gradient and total energy.
     * Child classes can override this to provide a more efficient implementation, than calculating
     * gradient and energy separately.
     * The implementation provided here is a fallback for derived classes and *not* more efficient than
     * separate calls.
     */
    virtual void Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalar & energy ) = 0;

    // Calculate the Energy contributions for the spins of a configuration
    virtual void Energy_Contributions_per_Spin(
        const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions ) = 0;

    // Calculate the total energy for a single spin
    virtual scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) = 0;

    // Hamiltonian name as string
    virtual const std::string & Name() const = 0;

    /*
     * Calculate the Hessian matrix of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient.
     */
    void Hessian_FD( const vectorfield & spins, MatrixX & hessian );

    /*
     * Calculate the energy gradient of a spin configuration.
     * This function uses finite differences and may thus be quite inefficient.
     */
    void Gradient_FD( const vectorfield & spins, vectorfield & gradient );

    // Calculate the Energy contributions for a spin configuration
    std::vector<std::pair<std::string, scalar>> Energy_Contributions( const vectorfield & spins );

    // Calculate the Energy of a spin configuration
    scalar Energy( const vectorfield & spins );

    std::size_t Number_of_Interactions() const;

    // Boundary conditions
    intfield boundary_conditions; // [3] (a, b, c)

protected:
    // Energy contributions per spin
    std::vector<std::pair<std::string, scalarfield>> energy_contributions_per_spin_;

private:
    scalar delta_;
};

} // namespace Engine

#endif
