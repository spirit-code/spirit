#ifndef SPIRIT_USE_CUDA

#include <engine/Hamiltonian_Heisenberg.hpp>
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
    // Construct a Heisenberg Hamiltonian with pairs
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalarfield mu_s,
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        pairfield exchange_pairs, scalarfield exchange_magnitudes,
        pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
        scalar ddi_radius,
        quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions), geometry(geometry),
        mu_s(mu_s),
        external_field_magnitude(external_field_magnitude * mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_pairs(exchange_pairs), exchange_magnitudes(exchange_magnitudes), exchange_n_shells(0),
        dmi_pairs(dmi_pairs), dmi_magnitudes(dmi_magnitudes), dmi_normals(dmi_normals), dmi_n_shells(0),
        quadruplets(quadruplets), quadruplet_magnitudes(quadruplet_magnitudes),
        ddi_cutoff_radius(ddi_radius)
    {
        #if defined SPIRIT_USE_OPENMP
        for (int i = 0; i < exchange_pairs.size(); ++i)
        {
            auto& p = exchange_pairs[i];
            auto& t = p.translations;
            this->exchange_pairs.push_back(Pair{p.j, p.i, {-t[0], -t[1], -t[2]}});
            this->exchange_magnitudes.push_back(exchange_magnitudes[i]);
        }
        for (int i = 0; i < dmi_pairs.size(); ++i)
        {
            auto& p = dmi_pairs[i];
            auto& t = p.translations;
            this->dmi_pairs.push_back(Pair{p.j, p.i, {-t[0], -t[1], -t[2]}});
            this->dmi_magnitudes.push_back(dmi_magnitudes[i]);
            this->dmi_normals.push_back(-dmi_normals[i]);
        }
        #endif

        // Generate DDI pairs, magnitudes, normals
        this->Update_DDI_Pairs();

        this->Update_Energy_Contributions();
    }

    // Construct a Heisenberg Hamiltonian from shells
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalarfield mu_s,
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        scalarfield exchange_magnitudes,
        scalarfield dmi_magnitudes, int dm_chirality,
        scalar ddi_radius,
        quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        mu_s(mu_s),
        external_field_magnitude(external_field_magnitude * mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_n_shells(exchange_magnitudes.size()),
        dmi_n_shells(dmi_magnitudes.size()),
        ddi_cutoff_radius(ddi_radius)
    {
        #if defined SPIRIT_USE_OPENMP
        // When parallelising (cuda or openmp), we need all neighbours per spin
        const bool remove_redundant = false;
        #else
        // When running on a single thread, we can ignore redundant neighbours
        const bool remove_redundant = true;
        #endif

        // Generate Exchange neighbours
        intfield exchange_shells(0);
        Neighbours::Get_Neighbours_in_Shells(*geometry, exchange_magnitudes.size(), exchange_pairs, exchange_shells, remove_redundant);
        for (unsigned int ipair = 0; ipair < exchange_pairs.size(); ++ipair)
        {
            this->exchange_magnitudes.push_back(exchange_magnitudes[exchange_shells[ipair]]);
        }

        // Generate DMI neighbours and normals
        intfield dmi_shells(0);
        Neighbours::Get_Neighbours_in_Shells(*geometry, dmi_magnitudes.size(), dmi_pairs, dmi_shells, remove_redundant);
        for (unsigned int ineigh = 0; ineigh < dmi_pairs.size(); ++ineigh)
        {
            this->dmi_normals.push_back(Neighbours::DMI_Normal_from_Pair(*geometry, dmi_pairs[ineigh], dm_chirality));
            this->dmi_magnitudes.push_back(dmi_magnitudes[dmi_shells[ineigh]]);
        }

        // Generate DDI pairs, magnitudes, normals
        this->Update_DDI_Pairs();

        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Heisenberg::Update_DDI_Pairs()
    {
        this->ddi_pairs      = Engine::Neighbours::Get_Pairs_in_Radius(*this->geometry, this->ddi_cutoff_radius);
        this->ddi_magnitudes = scalarfield(this->ddi_pairs.size());
        this->ddi_normals    = vectorfield(this->ddi_pairs.size());

        scalar magnitude;
        Vector3 normal;

        for (unsigned int i = 0; i < this->ddi_pairs.size(); ++i)
        {
            Engine::Neighbours::DDI_from_Pair(
                *this->geometry,
                { this->ddi_pairs[i].i, this->ddi_pairs[i].j, this->ddi_pairs[i].translations },
                this->ddi_magnitudes[i], this->ddi_normals[i]);
        }
    }

    void Hamiltonian_Heisenberg::Update_Energy_Contributions()
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
        if (this->exchange_pairs.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
            this->idx_exchange = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_exchange = -1;
        // DMI
        if (this->dmi_pairs.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
            this->idx_dmi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_dmi = -1;
        // Dipole-Dipole
        if (this->ddi_pairs.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"DD", scalarfield(0) });
            this->idx_ddi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_ddi = -1;
        // Quadruplets
        if (this->quadruplets.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"Quadruplets", scalarfield(0) });
            this->idx_quadruplet = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_quadruplet = -1;
    }

    void Hamiltonian_Heisenberg::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {
        if (contributions.size() != this->energy_contributions_per_spin.size())
        {
            contributions = this->energy_contributions_per_spin;
        }
        
        int nos = spins.size();
        for (auto& contrib : contributions)
        {
            // Allocate if not already allocated
            if (contrib.second.size() != nos) contrib.second = scalarfield(nos, 0);
            // Otherwise set to zero
            else Vectormath::fill(contrib.second, 0);
        }

        // External field
        if (this->idx_zeeman >=0 )     E_Zeeman(spins, contributions[idx_zeeman].second);

        // Anisotropy
        if (this->idx_anisotropy >=0 ) E_Anisotropy(spins, contributions[idx_anisotropy].second);

        // Exchange
        if (this->idx_exchange >=0 )   E_Exchange(spins, contributions[idx_exchange].second);
        // DMI
        if (this->idx_dmi >=0 )        E_DMI(spins,contributions[idx_dmi].second);
        // DD
        if (this->idx_ddi >=0 )        E_DDI(spins, contributions[idx_ddi].second);
        // Quadruplets
        if (this->idx_quadruplet >=0 ) E_Quadruplet(spins, contributions[idx_quadruplet].second);
    }

    void Hamiltonian_Heisenberg::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
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

    void Hamiltonian_Heisenberg::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
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

    void Hamiltonian_Heisenberg::E_Exchange(const vectorfield & spins, scalarfield & Energy)
    {
        #pragma omp parallel for
        for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair)
            {
                int ispin = exchange_pairs[i_pair].i + icell*geometry->n_cell_atoms;
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[i_pair]);
                if (jspin >= 0)
                {
                    Energy[ispin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin]);
                    #ifndef _OPENMP
                    Energy[jspin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin]);
                    #endif
                }
            }
        }
    }

    void Hamiltonian_Heisenberg::E_DMI(const vectorfield & spins, scalarfield & Energy)
    {
        #pragma omp parallel for
        for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair)
            {
                int ispin = dmi_pairs[i_pair].i + icell*geometry->n_cell_atoms;
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[i_pair]);
                if (jspin >= 0)
                {
                    Energy[ispin] -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin]));
                    #ifndef _OPENMP
                    Energy[jspin] -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin]));
                    #endif
                }
            }
        }
    }

    void Hamiltonian_Heisenberg::E_DDI(const vectorfield & spins, scalarfield & Energy)
    {
        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        const scalar mult = mu_0 * std::pow(mu_B, 2) / ( 4*Pi * 1e-30 );

        scalar result = 0.0;

        for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
        {
            if (ddi_magnitudes[i_pair] > 0.0)
            {
                for (int da = 0; da < geometry->n_cells[0]; ++da)
                {
                    for (int db = 0; db < geometry->n_cells[1]; ++db)
                    {
                        for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                        {
                            std::array<int, 3 > translations = { da, db, dc };
                            int i = ddi_pairs[i_pair].i;
                            int j = ddi_pairs[i_pair].j;
                            int ispin = i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
                            int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, ddi_pairs[i_pair]);
                            if (jspin >= 0)
                            {
                                Energy[ispin] -= 0.5 * this->mu_s[i] * this->mu_s[j] * mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
                                    (3 * spins[ispin].dot(ddi_normals[i_pair]) * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin].dot(spins[ispin]));
                                Energy[jspin] -= 0.5 * this->mu_s[i] * this->mu_s[j] * mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
                                    (3 * spins[ispin].dot(ddi_normals[i_pair]) * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin].dot(spins[ispin]));
                            }
                        }
                    }
                }
            }
        }
    }// end DipoleDipole


    void Hamiltonian_Heisenberg::E_Quadruplet(const vectorfield & spins, scalarfield & Energy)
    {
        for (unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad)
        {
            for (int da = 0; da < geometry->n_cells[0]; ++da)
            {
                for (int db = 0; db < geometry->n_cells[1]; ++db)
                {
                    for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        int ispin = quadruplets[iquad].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
                        int jspin = quadruplets[iquad].j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j);
                        int kspin = quadruplets[iquad].k + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k);
                        int lspin = quadruplets[iquad].l + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l);
                        
                        if ( check_atom_type(this->geometry->atom_types[ispin]) && check_atom_type(this->geometry->atom_types[jspin]) &&
                                check_atom_type(this->geometry->atom_types[kspin]) && check_atom_type(this->geometry->atom_types[lspin]) )
                        {
                            Energy[ispin] -= 0.25*quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * (spins[kspin].dot(spins[lspin]));
                            Energy[jspin] -= 0.25*quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * (spins[kspin].dot(spins[lspin]));
                            Energy[kspin] -= 0.25*quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * (spins[kspin].dot(spins[lspin]));
                            Energy[lspin] -= 0.25*quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * (spins[kspin].dot(spins[lspin]));
                        }
                    }
                }
            }
        }
    }



    void Hamiltonian_Heisenberg::Gradient(const vectorfield & spins, vectorfield & gradient)
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

        // Quadruplets
        this->Gradient_Quadruplet(spins, gradient);
    }

    void Hamiltonian_Heisenberg::Gradient_Zeeman(vectorfield & gradient)
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

    void Hamiltonian_Heisenberg::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
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

    void Hamiltonian_Heisenberg::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair)
            {
                int ispin = exchange_pairs[i_pair].i + icell*geometry->n_cell_atoms;
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[i_pair]);
                if (jspin >= 0)
                {
                    gradient[ispin] -= exchange_magnitudes[i_pair] * spins[jspin];
                    #ifndef _OPENMP
                    gradient[jspin] -= exchange_magnitudes[i_pair] * spins[ispin];
                    #endif
                }
            }
        }
    }

    void Hamiltonian_Heisenberg::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair)
            {
                int ispin = dmi_pairs[i_pair].i + icell*geometry->n_cell_atoms;
                int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[i_pair]);
                if (jspin >= 0)
                {
                    gradient[ispin] -= dmi_magnitudes[i_pair] * spins[jspin].cross(dmi_normals[i_pair]);
                    #ifndef _OPENMP
                    gradient[jspin] += dmi_magnitudes[i_pair] * spins[ispin].cross(dmi_normals[i_pair]);
                    #endif
                }
            }
        }
    }

    void Hamiltonian_Heisenberg::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
    {
        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        const scalar mult = mu_0 * std::pow(mu_B, 2) / ( 4*Pi * 1e-30 );
        
        for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
        {
            if (ddi_magnitudes[i_pair] > 0.0)
            {
                for (int da = 0; da < geometry->n_cells[0]; ++da)
                {
                    for (int db = 0; db < geometry->n_cells[1]; ++db)
                    {
                        for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                        {
                            scalar skalar_contrib = mult / std::pow(ddi_magnitudes[i_pair], 3.0);
                            std::array<int, 3 > translations = { da, db, dc };

                            int i = ddi_pairs[i_pair].i;
                            int j = ddi_pairs[i_pair].j;
                            int ispin = i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);	
                            int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, ddi_pairs[i_pair]);
                            if (jspin >= 0)
                            {
                                gradient[ispin] -= this->mu_s[j] * skalar_contrib * (3 * ddi_normals[i_pair] * spins[jspin].dot(ddi_normals[i_pair]) - spins[jspin]);
                                gradient[jspin] -= this->mu_s[i] * skalar_contrib * (3 * ddi_normals[i_pair] * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin]);
                            }
                        }
                    }
                }
            }
        }
    }//end Field_DipoleDipole


    void Hamiltonian_Heisenberg::Gradient_Quadruplet(const vectorfield & spins, vectorfield & gradient)
    {
        for (unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad)
        {
            int i = quadruplets[iquad].i;
            int j = quadruplets[iquad].j;
            int k = quadruplets[iquad].k;
            int l = quadruplets[iquad].l;
            for (int da = 0; da < geometry->n_cells[0]; ++da)
            {
                for (int db = 0; db < geometry->n_cells[1]; ++db)
                {
                    for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        int ispin = i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
                        int jspin = j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j);
                        int kspin = k + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k);
                        int lspin = l + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l);
                        
                        if ( check_atom_type(this->geometry->atom_types[ispin]) && check_atom_type(this->geometry->atom_types[jspin]) &&
                                check_atom_type(this->geometry->atom_types[kspin]) && check_atom_type(this->geometry->atom_types[lspin]) )
                        {
                            gradient[ispin] -= quadruplet_magnitudes[iquad] * spins[jspin] * (spins[kspin].dot(spins[lspin]));
                            gradient[jspin] -= quadruplet_magnitudes[iquad] * spins[ispin] * (spins[kspin].dot(spins[lspin]));
                            gradient[kspin] -= quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * spins[lspin];
                            gradient[lspin] -= quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * spins[kspin];
                        }
                    }
                }
            }
        }
    }


    void Hamiltonian_Heisenberg::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
        int nos = spins.size();

        // Set to zero
        hessian.setZero();

        // Single Spin elements
        for (int da = 0; da < geometry->n_cells[0]; ++da)
        {
            for (int db = 0; db < geometry->n_cells[1]; ++db)
            {
                for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                {
                    std::array<int, 3 > translations = { da, db, dc };
                    int icell = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
                    for (int alpha = 0; alpha < 3; ++alpha)
                    {
                        for ( int beta = 0; beta < 3; ++beta )
                        {
                            for (unsigned int i = 0; i < anisotropy_indices.size(); ++i)
                            {
                                if ( check_atom_type(this->geometry->atom_types[anisotropy_indices[i]]) )
                                {
                                    int idx_i = 3 * icell + anisotropy_indices[i] + alpha;
                                    int idx_j = 3 * icell + anisotropy_indices[i] + beta;
                                    // scalar x = -2.0*this->anisotropy_magnitudes[i] * std::pow(this->anisotropy_normals[i][alpha], 2);
                                    hessian( idx_i, idx_j ) += -2.0 * this->anisotropy_magnitudes[i] * 
                                                                    this->anisotropy_normals[i][alpha] * 
                                                                    this->anisotropy_normals[i][beta];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Spin Pair elements
        // Exchange
        for (int da = 0; da < geometry->n_cells[0]; ++da)
        {
            for (int db = 0; db < geometry->n_cells[1]; ++db)
            {
                for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                {
                    std::array<int, 3 > translations = { da, db, dc };
                    for (unsigned int i_pair = 0; i_pair < this->exchange_pairs.size(); ++i_pair)
                    {
                        int ispin = exchange_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[i_pair]);
                        if (jspin >= 0)
                        {
                            for (int alpha = 0; alpha < 3; ++alpha)
                            {
                                int i = 3 * ispin + alpha;
                                int j = 3 * jspin + alpha;

                                hessian(i, j) += -exchange_magnitudes[i_pair];
                                #ifndef _OPENMP
                                hessian(j, i) += -exchange_magnitudes[i_pair];
                                #endif
                            }
                        }
                    }
                }
            }
        }

        // DMI
        for (int da = 0; da < geometry->n_cells[0]; ++da)
        {
            for (int db = 0; db < geometry->n_cells[1]; ++db)
            {
                for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                {
                    std::array<int, 3 > translations = { da, db, dc };
                    for (unsigned int i_pair = 0; i_pair < this->dmi_pairs.size(); ++i_pair)
                    {
                        int ispin = dmi_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[i_pair]);
                        if (jspin >= 0)
                        {
                            int i = 3*ispin;
                            int j = 3*jspin;

                            hessian(i+2, j+1) +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                            hessian(i+1, j+2) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                            hessian(i, j+2)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                            hessian(i+2, j)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                            hessian(i+1, j)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                            hessian(i, j+1)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];

                            #ifndef _OPENMP
                            hessian(j+1, i+2) +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                            hessian(j+2, i+1) += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                            hessian(j+2, i)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                            hessian(j, i+2)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                            hessian(j, i+1)   +=  dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                            hessian(j+1, i)   += -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                            #endif
                        }
                    }
                }
            }
        }

        //// Dipole-Dipole
        //for (unsigned int i_pair = 0; i_pair < this->DD_indices.size(); ++i_pair)
        //{
        //	// indices
        //	int idx_1 = DD_indices[i_pair][0];
        //	int idx_2 = DD_indices[i_pair][1];
        //	// prefactor
        //	scalar prefactor = 0.0536814951168
        //		* this->mu_s[idx_1] * this->mu_s[idx_2]
        //		/ std::pow(DD_magnitude[i_pair], 3);
        //	// components
        //	for (int alpha = 0; alpha < 3; ++alpha)
        //	{
        //		for (int beta = 0; beta < 3; ++beta)
        //		{
        //			int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
        //			if (alpha == beta)
        //				hessian[idx_h] += prefactor;
        //			hessian[idx_h] += -3.0*prefactor*DD_normal[i_pair][alpha] * DD_normal[i_pair][beta];
        //		}
        //	}
        //}

        // Quadruplets
    }

    // Hamiltonian name as string
    static const std::string name = "Heisenberg";
    const std::string& Hamiltonian_Heisenberg::Name() { return name; }
}

#endif