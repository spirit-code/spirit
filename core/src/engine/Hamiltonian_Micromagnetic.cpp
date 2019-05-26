#ifndef SPIRIT_USE_CUDA

#include <engine/Hamiltonian_Micromagnetic.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Core>


using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;
using Engine::Vectormath::idx_from_tupel;


namespace Engine
{
    Hamiltonian_Micromagnetic::Hamiltonian_Micromagnetic(
        scalar external_field_magnitude, Vector3 external_field_normal,
        Matrix3 anisotropy_tensor,
        Matrix3 exchange_tensor,
        Matrix3 dmi_tensor,
        std::shared_ptr<Data::Geometry> geometry,
        int spatial_gradient_order,
        intfield boundary_conditions
    ) : Hamiltonian(boundary_conditions), spatial_gradient_order(spatial_gradient_order), geometry(geometry),
        external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
        anisotropy_tensor(anisotropy_tensor)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Interactions();
    }

    Hamiltonian_Micromagnetic::Hamiltonian_Micromagnetic(
        scalar external_field_magnitude, Vector3 external_field_normal,
        Matrix3 anisotropy_tensor,
        scalar exchange_constant,
        scalar dmi_constant,
        std::shared_ptr<Data::Geometry> geometry,
        int spatial_gradient_order,
        intfield boundary_conditions
    ) : Hamiltonian(boundary_conditions), spatial_gradient_order(spatial_gradient_order), geometry(geometry),
        external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
        anisotropy_tensor(anisotropy_tensor)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Interactions();
    }

    void Hamiltonian_Micromagnetic::Update_Interactions()
    {
        #if defined(SPIRIT_USE_OPENMP)
        // When parallelising (cuda or openmp), we need all neighbours per spin
        const bool use_redundant_neighbours = true;
        #else
        // When running on a single thread, we can ignore redundant neighbours
        const bool use_redundant_neighbours = false;
        #endif

        // TODO: make sure that the geometry can be treated with this model:
        //       - rectilinear, only one "atom" per cell
        // if( geometry->n_cell_atoms != 1 )
        //     Log(...)

        // TODO: generate neighbour information for pairwise interactions

        // TODO: prepare dipolar interactions

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Micromagnetic::Update_Energy_Contributions()
    {
        this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

        // External field
        if( this->external_field_magnitude > 0 )
        {
            this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
            this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
        }
        else
            this->idx_zeeman = -1;
        // TODO: Anisotropy
        // if( ... )
        // {
        //     this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0) });
        //     this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
        // }
        // else
            this->idx_anisotropy = -1;
        // TODO: Exchange
        // if( ... )
        // {
        //     this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
        //     this->idx_exchange = this->energy_contributions_per_spin.size()-1;
        // }
        // else
            this->idx_exchange = -1;
        // TODO: DMI
        // if( ... )
        // {
        //     this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
        //     this->idx_dmi = this->energy_contributions_per_spin.size()-1;
        // }
        // else
            this->idx_dmi = -1;
        // TODO: DDI
        // if( ... )
        // {
        //     this->energy_contributions_per_spin.push_back({"DDI", scalarfield(0) });
        //     this->idx_ddi = this->energy_contributions_per_spin.size()-1;
        // }
        // else
            this->idx_ddi = -1;
    }

    void Hamiltonian_Micromagnetic::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {
        if( contributions.size() != this->energy_contributions_per_spin.size() )
        {
            contributions = this->energy_contributions_per_spin;
        }

        int nos = spins.size();
        for( auto& contrib : contributions )
        {
            // Allocate if not already allocated
            if (contrib.second.size() != nos) contrib.second = scalarfield(nos, 0);
            // Otherwise set to zero
            else Vectormath::fill(contrib.second, 0);
        }

        // External field
        if( this->idx_zeeman >=0 )     E_Zeeman(spins, contributions[idx_zeeman].second);

        // Anisotropy
        if( this->idx_anisotropy >=0 ) E_Anisotropy(spins, contributions[idx_anisotropy].second);

        // Exchange
        if( this->idx_exchange >=0 )   E_Exchange(spins, contributions[idx_exchange].second);
        // DMI
        if( this->idx_dmi >=0 )        E_DMI(spins,contributions[idx_dmi].second);
    }

    void Hamiltonian_Micromagnetic::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        auto& mu_s = this->geometry->mu_s;

        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            if( check_atom_type(this->geometry->atom_types[icell]) )
                Energy[icell] -= mu_s[icell] * this->external_field_magnitude * this->external_field_normal.dot(spins[icell]);
        }
    }

    void Hamiltonian_Micromagnetic::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
    {
    }

    void Hamiltonian_Micromagnetic::E_Exchange(const vectorfield & spins, scalarfield & Energy)
    {
    }

    void Hamiltonian_Micromagnetic::E_DMI(const vectorfield & spins, scalarfield & Energy)
    {
    }

    void Hamiltonian_Micromagnetic::E_DDI(const vectorfield & spins, scalarfield & Energy)
    {
    }


    scalar Hamiltonian_Micromagnetic::Energy_Single_Spin(int ispin, const vectorfield & spins)
    {
        scalar Energy = 0;
        return Energy;
    }


    void Hamiltonian_Micromagnetic::Gradient(const vectorfield & spins, vectorfield & gradient)
    {
        // Set to zero
        Vectormath::fill(gradient, {0,0,0});

        // External field
        this->Gradient_Zeeman(gradient);

        // Anisotropy
        this->Gradient_Anisotropy(spins, gradient);

        // Exchange
        this->Gradient_Exchange(spins, gradient);

        // DMI
        this->Gradient_DMI(spins, gradient);
    }


    void Hamiltonian_Micromagnetic::Gradient_Zeeman(vectorfield & gradient)
    {
        auto& mu_s = this->geometry->mu_s;

        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            if( check_atom_type(this->geometry->atom_types[icell]) )
                gradient[icell] -= mu_s[icell] * this->external_field_magnitude * this->external_field_normal;
        }
    }

    void Hamiltonian_Micromagnetic::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
    }

    void Hamiltonian_Micromagnetic::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
    }

    void Hamiltonian_Micromagnetic::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
    }


    void Hamiltonian_Micromagnetic::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
    }


    // Hamiltonian name as string
    static const std::string name = "Micromagnetic";
    const std::string& Hamiltonian_Micromagnetic::Name() { return name; }
}

#endif