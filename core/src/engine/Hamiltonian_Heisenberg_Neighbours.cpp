#ifndef USE_CUDA

#include <Spirit_Defines.h>
#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Dense>

using namespace Data;
using namespace Utility;
using Utility::Constants::mu_B;
using Utility::Constants::mu_0;
using Utility::Constants::Pi;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;

namespace Engine
{
    Hamiltonian_Heisenberg_Neighbours::Hamiltonian_Heisenberg_Neighbours(
        scalarfield mu_s,
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        scalarfield exchange_magnitudes,
        scalarfield dmi_magnitudes, int dm_chirality,
        scalar ddi_radius,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        mu_s(mu_s),
        external_field_magnitude(external_field_magnitude * mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_magnitudes(exchange_magnitudes),
        dmi_magnitudes(dmi_magnitudes),
        ddi_radius(ddi_radius)
    {
        // Generate Exchange neighbours
        exchange_neighbours = Neighbours::Get_Neighbours_in_Shells(*geometry, exchange_magnitudes.size());

        // Generate DMI neighbours and normals
        dmi_neighbours = Neighbours::Get_Neighbours_in_Shells(*geometry, dmi_magnitudes.size());
        for (unsigned int ineigh = 0; ineigh < dmi_neighbours.size(); ++ineigh)
        {
            dmi_normals.push_back(Neighbours::DMI_Normal_from_Pair(*geometry, dmi_neighbours[ineigh], dm_chirality));
        }

        // Generate DDI neighbours, magnitudes and normals
        this->ddi_neighbours = Engine::Neighbours::Get_Neighbours_in_Radius(*this->geometry, ddi_radius);
        scalar magnitude;
        Vector3 normal;
        for (unsigned int i=0; i<ddi_neighbours.size(); ++i)
        {
            Engine::Neighbours::DDI_from_Pair(*this->geometry, ddi_neighbours[i], magnitude, normal);
            this->ddi_magnitudes.push_back(magnitude);
            this->ddi_normals.push_back(normal);
        }

        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Heisenberg_Neighbours::Update_N_Neighbour_Shells(int n_shells_exchange, int n_shells_dmi)
    {
        if (this->exchange_magnitudes.size() != n_shells_exchange)
        {
            this->exchange_magnitudes = scalarfield(n_shells_exchange);
            // Re-calculate exchange neighbour list
        }
        if (this->dmi_magnitudes.size() != n_shells_dmi)
        {
            this->dmi_magnitudes = scalarfield(n_shells_dmi);
            // Re-calculate dmi neighbour list
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::Update_Energy_Contributions()
    {
        this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

        // External field
        if (this->external_field_magnitude > 0)
        {
            this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
            this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_zeeman = -1;
        // Anisotropy
        if (this->anisotropy_indices.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0) });
            this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_anisotropy = -1;
        // Exchange
        if (this->exchange_neighbours.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
            this->idx_exchange = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_exchange = -1;
        // DMI
        if (this->dmi_neighbours.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
            this->idx_dmi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_dmi = -1;
        // Dipole-Dipole
        if (this->ddi_neighbours.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"DD", scalarfield(0) });
            this->idx_ddi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_ddi = -1;
    }

    void Hamiltonian_Heisenberg_Neighbours::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {
        if (contributions.size() != this->energy_contributions_per_spin.size())
        {
            contributions = this->energy_contributions_per_spin;
        }

        int nos = spins.size();
        for (auto& pair : energy_contributions_per_spin)
        {
            // Allocate if not already allocated
            if (pair.second.size() != nos) pair.second = scalarfield(nos, 0);
            // Otherwise set to zero
            else for (auto& pair : energy_contributions_per_spin) Vectormath::fill(pair.second, 0);
        }

        // External field
        if (this->idx_zeeman >=0 )     E_Zeeman(spins, energy_contributions_per_spin[idx_zeeman].second);
        // Anisotropy
        if (this->idx_anisotropy >=0 ) E_Anisotropy(spins, energy_contributions_per_spin[idx_anisotropy].second);

        // Exchange
        if (this->idx_exchange >=0 )   E_Exchange(spins,energy_contributions_per_spin[idx_exchange].second);
        // DMI
        if (this->idx_dmi >=0 )        E_DMI(spins, energy_contributions_per_spin[idx_dmi].second);
        // DDI
        if (this->idx_ddi >=0 )        E_DDI(spins, energy_contributions_per_spin[idx_ddi].second);
    }

    void Hamiltonian_Heisenberg_Neighbours::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        const int N = geometry->n_cell_atoms;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int ibasis = 0; ibasis < N; ++ibasis)
            {
                int ispin = icell*N + ibasis;
                if (check_atom_type(this->geometry->atom_types[ispin]))
                    Energy[ispin] -= this->mu_s[ibasis] * this->external_field_magnitude * this->external_field_normal.dot(spins[ispin]);
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
    {
        const int N = geometry->n_cell_atoms;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int iani = 0; iani < anisotropy_indices.size(); ++iani)
            {
                int ispin = icell*N + anisotropy_indices[iani];
                if (check_atom_type(this->geometry->atom_types[ispin]))
                    Energy[ispin] -= this->anisotropy_magnitudes[iani] * std::pow(anisotropy_normals[iani].dot(spins[ispin]), 2.0);
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::E_Exchange(const vectorfield & spins, scalarfield & Energy)
    {
        #pragma omp parallel for
        for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
        {
            // auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
            for (unsigned int ineigh = 0; ineigh < exchange_neighbours.size(); ++ineigh)
            {
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_neighbours[ineigh]);
                if ( jspin >= 0 )
                {
                    auto& ishell = exchange_neighbours[ineigh].idx_shell;
                    Energy[ispin] -= 0.5 * exchange_magnitudes[ishell] * spins[ispin].dot(spins[jspin]);
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::E_DMI(const vectorfield & spins, scalarfield & Energy)
    {
        #pragma omp parallel for
        for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
        {
            for (unsigned int ineigh = 0; ineigh < dmi_neighbours.size(); ++ineigh)
            {
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_neighbours[ineigh]);
                if ( jspin >= 0 )
                {
                    auto& ishell = dmi_neighbours[ineigh].idx_shell;
                    Energy[ispin] -= 0.5 * dmi_magnitudes[ishell] * dmi_normals[ineigh].dot(spins[ispin].cross(spins[jspin]));
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::E_DDI(const vectorfield & spins, scalarfield & Energy)
    {
        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        const scalar mult = mu_0 * std::pow(mu_B, 2) / ( 4*Pi * 1e-30 );

        scalar result = 0.0;
        const int N = geometry->n_cell_atoms;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int ibasis = 0; ibasis < N; ++ibasis)
            {
                int ispin = icell * N + ibasis;
                for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
                {
                    if (ddi_magnitudes[ineigh] > 0.0)
                    {
                        int jbasis = ddi_neighbours[ineigh].j;
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, ddi_neighbours[ineigh]);
                        if (jspin >= 0)
                        {
                            Energy[ispin] -= 0.5 * this->mu_s[ibasis] * this->mu_s[jbasis] * mult / std::pow(ddi_magnitudes[ineigh], 3.0) *
                                (3 * spins[jspin].dot(ddi_normals[ineigh]) * spins[ispin].dot(ddi_normals[ineigh]) - spins[ispin].dot(spins[jspin]));
                            Energy[jspin] -= 0.5 * this->mu_s[ibasis] * this->mu_s[jbasis] * mult / std::pow(ddi_magnitudes[ineigh], 3.0) *
                                (3 * spins[jspin].dot(ddi_normals[ineigh]) * spins[ispin].dot(ddi_normals[ineigh]) - spins[ispin].dot(spins[jspin]));
                        }
                    }
                }
            }
        }
    }// end DipoleDipole



    void Hamiltonian_Heisenberg_Neighbours::Gradient(const vectorfield & spins, vectorfield & gradient)
    {
        // Set to zero
        Vectormath::fill(gradient, {0,0,0});

        // External field
        Gradient_Zeeman(gradient);

        // Anisotropy
        Gradient_Anisotropy(spins, gradient);

        // Exchange
        this->Gradient_Exchange(spins, gradient);
        // DMI
        this->Gradient_DMI(spins, gradient);
        // DD
        this->Gradient_DDI(spins, gradient);
    }

    void Hamiltonian_Heisenberg_Neighbours::Gradient_Zeeman(vectorfield & gradient)
    {
        const int N = geometry->n_cell_atoms;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int ibasis = 0; ibasis < N; ++ibasis)
            {
                int ispin = icell*N + ibasis;
                if (check_atom_type(this->geometry->atom_types[ispin]))
                    gradient[ispin] -= this->mu_s[ibasis] * this->external_field_magnitude * this->external_field_normal;
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
        const int N = geometry->n_cell_atoms;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int iani = 0; iani < anisotropy_indices.size(); ++iani)
            {
                int ispin = icell*N + anisotropy_indices[iani];
                if (check_atom_type(this->geometry->atom_types[ispin]))
                    gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[iani] * this->anisotropy_normals[iani] * anisotropy_normals[iani].dot(spins[ispin]);
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
        {
            for (unsigned int ineigh = 0; ineigh < exchange_neighbours.size(); ++ineigh)
            {
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_neighbours[ineigh]);
                if ( jspin >= 0 )
                {
                    auto& ishell = exchange_neighbours[ineigh].idx_shell;
                    gradient[ispin] -= exchange_magnitudes[ishell] * spins[jspin];
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
        {
            auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
            for (unsigned int ineigh = 0; ineigh < dmi_neighbours.size(); ++ineigh)
            {
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_neighbours[ineigh]);
                if ( jspin >= 0 )
                {
                    auto& ishell = dmi_neighbours[ineigh].idx_shell;
                    gradient[ispin] -= dmi_magnitudes[ishell] * spins[jspin].cross(dmi_normals[ineigh]);
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Neighbours::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
    {
        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        const scalar mult = mu_0 * std::pow(mu_B, 2) / ( 4*Pi * 1e-30 );

        const int N = geometry->n_cell_atoms;
        
        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int ibasis = 0; ibasis < N; ++ibasis)
            {
                int ispin = icell * N + ibasis;
                for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
                {
                    if (ddi_magnitudes[ineigh] > 0.0)
                    {
                        int jbasis = ddi_neighbours[ineigh].j;
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, ddi_neighbours[ineigh]);
                        if (jspin >= 0)
                        {
                            scalar skalar_contrib = mult / std::pow(ddi_magnitudes[ineigh], 3.0);
                            gradient[ispin] -= this->mu_s[jbasis] * skalar_contrib * (3 * ddi_normals[ineigh] * spins[jspin].dot(ddi_normals[ineigh]) - spins[jspin]);
                            gradient[jspin] -= this->mu_s[ibasis] * skalar_contrib * (3 * ddi_normals[ineigh] * spins[ispin].dot(ddi_normals[ineigh]) - spins[ispin]);
                        }
                    }
                }
            }
        }
    }//end Field_DipoleDipole


    void Hamiltonian_Heisenberg_Neighbours::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
        int nos = spins.size();

        // Set to zero
        // for (auto& h : hessian) h = 0;
        hessian.setZero();

        // Single Spin elements
        for (int alpha = 0; alpha < 3; ++alpha)
        {
            for (unsigned int i = 0; i < anisotropy_indices.size(); ++i)
            {
                int idx = anisotropy_indices[i];
                // scalar x = -2.0*this->anisotropy_magnitudes[i] * std::pow(this->anisotropy_normals[i][alpha], 2);
                hessian(3*idx + alpha, 3*idx + alpha) += -2.0*this->anisotropy_magnitudes[i]*std::pow(this->anisotropy_normals[i][alpha],2);
            }
        }

        // std::cerr << "calculated hessian" << std::endl;

        // Spin Pair elements
        // Exchange
        for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
        {
            auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
            for (unsigned int ineigh = 0; ineigh < this->exchange_neighbours.size(); ++ineigh)
            {
                for (int alpha = 0; alpha < 3; ++alpha)
                {
                    //int idx_i = 3 * exchange_neighbours[i_pair][0] + alpha;
                    //int idx_j = 3 * exchange_neighbours[i_pair][1] + alpha;
                    int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, exchange_neighbours[ineigh].translations);
                    int ishell = exchange_neighbours[ineigh].idx_shell;
                    hessian(ispin, jspin) += -exchange_magnitudes[ineigh];
                    hessian(jspin, ispin) += -exchange_magnitudes[ineigh];
                }
            }
        }
        // DMI
        for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
        {
            auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
            for (unsigned int ineigh = 0; ineigh < this->dmi_neighbours.size(); ++ineigh)
            {
                for (int alpha = 0; alpha < 3; ++alpha)
                {
                    for (int beta = 0; beta < 3; ++beta)
                    {
                        int idx_i = 3 * dmi_neighbours[ineigh].i + alpha;
                        int idx_j = 3 * dmi_neighbours[ineigh].j + beta;
                        if ((alpha == 0 && beta == 1))
                        {
                            hessian(idx_i, idx_j) +=
                                -dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
                            hessian(idx_j, idx_i) +=
                                -dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
                        }
                        else if ((alpha == 1 && beta == 0))
                        {
                            hessian(idx_i, idx_j) +=
                                dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
                            hessian(idx_j, idx_i) +=
                                dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
                        }
                        else if ((alpha == 0 && beta == 2))
                        {
                            hessian(idx_i, idx_j) +=
                                dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
                            hessian(idx_j, idx_i) +=
                                dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
                        }
                        else if ((alpha == 2 && beta == 0))
                        {
                            hessian(idx_i, idx_j) +=
                                -dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
                            hessian(idx_j, idx_i) +=
                                -dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
                        }
                        else if ((alpha == 1 && beta == 2))
                        {
                            hessian(idx_i, idx_j) +=
                                -dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
                            hessian(idx_j, idx_i) +=
                                -dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
                        }
                        else if ((alpha == 2 && beta == 1))
                        {
                            hessian(idx_i, idx_j) +=
                                dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
                            hessian(idx_j, idx_i) +=
                                dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
                        }
                    }
                }
            }
        }
        //// Dipole-Dipole
        //for (unsigned int i_pair = 0; i_pair < this->ddi_neighbours[i_periodicity].size(); ++i_pair)
        //{
        //	// indices
        //	int idx_1 = ddi_neighbours[i_periodicity][i_pair][0];
        //	int idx_2 = ddi_neighbours[i_periodicity][i_pair][1];
        //	// prefactor
        //	scalar prefactor = 0.0536814951168
        //		* this->mu_s[idx_1] * this->mu_s[idx_2]
        //		/ std::pow(ddi_magnitude[i_periodicity][i_pair], 3);
        //	// components
        //	for (int alpha = 0; alpha < 3; ++alpha)
        //	{
        //		for (int beta = 0; beta < 3; ++beta)
        //		{
        //			int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
        //			if (alpha == beta)
        //				hessian[idx_h] += prefactor;
        //			hessian[idx_h] += -3.0*prefactor*DD_normal[i_periodicity][i_pair][alpha] * DD_normal[i_periodicity][i_pair][beta];
        //		}
        //	}
        //}
    }

    // Hamiltonian name as string
    static const std::string name = "Heisenberg (Neighbours)";
    const std::string& Hamiltonian_Heisenberg_Neighbours::Name() { return name; }
}

#endif