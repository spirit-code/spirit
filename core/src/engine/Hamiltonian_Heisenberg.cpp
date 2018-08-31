#ifndef SPIRIT_USE_CUDA

#include <engine/Hamiltonian_Heisenberg.hpp>
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
    // Construct a Heisenberg Hamiltonian with pairs
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        pairfield exchange_pairs, scalarfield exchange_magnitudes,
        pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
        scalar ddi_radius,
        quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        external_field_magnitude(external_field_magnitude * C::mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_pairs_in(exchange_pairs), exchange_magnitudes_in(exchange_magnitudes), exchange_shell_magnitudes(0),
        dmi_pairs_in(dmi_pairs), dmi_magnitudes_in(dmi_magnitudes), dmi_normals_in(dmi_normals), dmi_shell_magnitudes(0), dmi_shell_chirality(0),
        quadruplets(quadruplets), quadruplet_magnitudes(quadruplet_magnitudes),
        ddi_cutoff_radius(ddi_radius)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Interactions();
    }

    // Construct a Heisenberg Hamiltonian from shells
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        scalarfield exchange_shell_magnitudes,
        scalarfield dmi_shell_magnitudes, int dm_chirality,
        scalar ddi_radius,
        quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        external_field_magnitude(external_field_magnitude * C::mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_pairs_in(0), exchange_magnitudes_in(0), exchange_shell_magnitudes(exchange_shell_magnitudes),
        dmi_pairs_in(0), dmi_magnitudes_in(0), dmi_normals_in(0), dmi_shell_magnitudes(dmi_shell_magnitudes), dmi_shell_chirality(dm_chirality),
        quadruplets(quadruplets), quadruplet_magnitudes(quadruplet_magnitudes),
        ddi_cutoff_radius(ddi_radius)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Interactions();
    }

    void Hamiltonian_Heisenberg::Update_Interactions()
    {
        #if defined(SPIRIT_USE_OPENMP)
        // When parallelising (cuda or openmp), we need all neighbours per spin
        const bool use_redundant_neighbours = true;
        #else
        // When running on a single thread, we can ignore redundant neighbours
        const bool use_redundant_neighbours = false;
        #endif

        // Exchange
        this->exchange_pairs      = pairfield(0);
        this->exchange_magnitudes = scalarfield(0);
        if( exchange_shell_magnitudes.size() > 0 )
        {
            // Generate Exchange neighbours
            intfield exchange_shells(0);
            Neighbours::Get_Neighbours_in_Shells(*geometry, exchange_shell_magnitudes.size(), exchange_pairs, exchange_shells, use_redundant_neighbours);
            for (unsigned int ipair = 0; ipair < exchange_pairs.size(); ++ipair)
            {
                this->exchange_magnitudes.push_back(exchange_shell_magnitudes[exchange_shells[ipair]]);
            }
        }
        else
        {
            // Use direct list of pairs
            this->exchange_pairs      = this->exchange_pairs_in;
            this->exchange_magnitudes = this->exchange_magnitudes_in;
            if( use_redundant_neighbours )
            {
                for (int i = 0; i < exchange_pairs_in.size(); ++i)
                {
                    auto& p = exchange_pairs_in[i];
                    auto& t = p.translations;
                    this->exchange_pairs.push_back(Pair{p.j, p.i, {-t[0], -t[1], -t[2]}});
                    this->exchange_magnitudes.push_back(exchange_magnitudes_in[i]);
                }
            }
        }

        // DMI
        this->dmi_pairs      = pairfield(0);
        this->dmi_magnitudes = scalarfield(0);
        this->dmi_normals    = vectorfield(0);
        if( dmi_shell_magnitudes.size() > 0 )
        {
            // Generate DMI neighbours and normals
            intfield dmi_shells(0);
            Neighbours::Get_Neighbours_in_Shells(*geometry, dmi_shell_magnitudes.size(), dmi_pairs, dmi_shells, use_redundant_neighbours);
            for (unsigned int ineigh = 0; ineigh < dmi_pairs.size(); ++ineigh)
            {
                this->dmi_normals.push_back(Neighbours::DMI_Normal_from_Pair(*geometry, dmi_pairs[ineigh], this->dmi_shell_chirality));
                this->dmi_magnitudes.push_back(dmi_shell_magnitudes[dmi_shells[ineigh]]);
            }
        }
        else
        {
            // Use direct list of pairs
            this->dmi_pairs      = this->dmi_pairs_in;
            this->dmi_magnitudes = this->dmi_magnitudes_in;
            this->dmi_normals    = this->dmi_normals_in;
            if( use_redundant_neighbours )
            {
                for (int i = 0; i < dmi_pairs_in.size(); ++i)
                {
                    auto& p = dmi_pairs_in[i];
                    auto& t = p.translations;
                    this->dmi_pairs.push_back(Pair{p.j, p.i, {-t[0], -t[1], -t[2]}});
                    this->dmi_magnitudes.push_back(dmi_magnitudes_in[i]);
                    this->dmi_normals.push_back(-dmi_normals_in[i]);
                }
            }
        }

        // Dipole-dipole
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

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Heisenberg::Update_Energy_Contributions()
    {
        Prepare_DDI();
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
        if (this->ddi_pairs.size() > 0 || true) //todo: get rid of ||true
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
        if (this->idx_ddi >=0 )    
            // E_DDI(spins, contributions[idx_ddi].second);
            E_DDI_FFT(spins, contributions[idx_ddi].second);
        // Quadruplets
        if (this->idx_quadruplet >=0 ) E_Quadruplet(spins, contributions[idx_quadruplet].second);
    }

    void Hamiltonian_Heisenberg::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        const int N = geometry->n_cell_atoms;
        auto& mu_s = this->geometry->mu_s;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int ibasis = 0; ibasis < N; ++ibasis)
            {
                int ispin = icell*N + ibasis;
                if( check_atom_type(this->geometry->atom_types[ispin]) )
                    Energy[ispin] -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal.dot(spins[ispin]);
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
        auto& mu_s = this->geometry->mu_s;
        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        const scalar mult = C::mu_0 * std::pow(C::mu_B, 2) / ( 4*C::Pi * 1e-30 );

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
                                Energy[ispin] -= 0.5 * mu_s[ispin] * mu_s[jspin] * mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
                                    (3 * spins[ispin].dot(ddi_normals[i_pair]) * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin].dot(spins[ispin]));
                                Energy[jspin] -= 0.5 * mu_s[ispin] * mu_s[jspin] * mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
                                    (3 * spins[ispin].dot(ddi_normals[i_pair]) * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin].dot(spins[ispin]));
                            }
                        }
                    }
                }
            }
        }
    }// end DipoleDipole

    void Hamiltonian_Heisenberg::E_DDI_FFT(const vectorfield & spins, scalarfield & Energy)
    {
        scalar Energy_DDI = 0;
        vectorfield gradients_temp;
        gradients_temp.resize(geometry->nos);
        Vectormath::fill(gradients_temp, {0,0,0});
        Gradient_DDI_FFT(spins, gradients_temp);

        // === DEBUG: begin gradient comparison ===
            // vectorfield gr./sadients_temp_dir;
            // gradients_temp_dir.resize(this->geometry->nos);
            // Vectormath::fill(gradients_temp_dir, {0,0,0});
            // Gradient_DDI_direct(spins, gradients_temp_dir);

            // //get deviation
            // std::array<scalar, 3> deviation = {0,0,0};
            // std::array<scalar, 3> avg = {0,0,0};
            // for(int i = 0; i < this->geometry->nos; i++)
            // {
            //     for(int d = 0; d < 3; d++)
            //     {
            //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
            //         avg[d] += gradients_temp_dir[i][d];
            //     }
            // }
            // std::cerr << "Avg. Gradient = " << avg[0]/this->geometry->nos << " " << avg[1]/this->geometry->nos << " " << avg[2]/this->geometry->nos << std::endl;
            // std::cerr << "Avg. Deviation = " << deviation[0]/this->geometry->nos << " " << deviation[1]/this->geometry->nos << " " << deviation[2]/this->geometry->nos << std::endl;
        //==== DEBUG: end gradient comparison ====

        for(int ib = 0; ib < this->geometry->n_cell_atoms; ib++)
        {
            auto mu = geometry->mu_s[ib];
            for (int i_cell = 0; i_cell < geometry->n_cells_total; i_cell++)
            {
                auto idx = ib + i_cell * this->geometry->n_cell_atoms;
                Energy[idx] += 0.5 * mu * spins[idx].dot(gradients_temp[idx]);
                Energy_DDI += 0.5 * mu * spins[idx].dot(gradients_temp[idx]);
            }
        }
    }

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


    scalar Hamiltonian_Heisenberg::Energy_Single_Spin(int ispin_in, const vectorfield & spins)
    {
        scalar Energy = 0;
        if( check_atom_type(this->geometry->atom_types[ispin_in]) )
        {
            int icell  = ispin_in / this->geometry->n_cell_atoms;
            int ibasis = ispin_in - icell*this->geometry->n_cell_atoms;
            auto& mu_s = this->geometry->mu_s;

            // External field
            if (this->idx_zeeman >= 0)
                Energy -= mu_s[ispin_in] * this->external_field_magnitude * this->external_field_normal.dot(spins[ispin_in]);

            // Anisotropy
            if (this->idx_anisotropy >= 0)
            {
                for (int iani = 0; iani < anisotropy_indices.size(); ++iani)
                {
                    if (anisotropy_indices[iani] == ibasis)
                    {
                        if (check_atom_type(this->geometry->atom_types[ispin_in]))
                            Energy -= this->anisotropy_magnitudes[iani] * std::pow(anisotropy_normals[iani].dot(spins[ispin_in]), 2.0);
                    }
                }
            }

            // Exchange
            if (this->idx_exchange >= 0)
            {
                for (unsigned int ipair = 0; ipair < exchange_pairs.size(); ++ipair)
                {
                    if (exchange_pairs[ipair].i == ibasis)
                    {
                        int ispin = exchange_pairs[ipair].i + icell*geometry->n_cell_atoms;
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[ipair]);
                        if (jspin >= 0)
                        {
                            Energy -= 0.5 * this->exchange_magnitudes[ipair] * spins[ispin].dot(spins[jspin]);
                        }
                        #ifndef _OPENMP
                        jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, exchange_pairs[ipair], true);
                        if (jspin >= 0)
                        {
                            Energy -= 0.5 * this->exchange_magnitudes[ipair] * spins[ispin].dot(spins[jspin]);
                        }
                        #endif
                    }
                }
            }

            // DMI
            if (this->idx_dmi >= 0)
            {
                for (unsigned int ipair = 0; ipair < dmi_pairs.size(); ++ipair)
                {
                    if (dmi_pairs[ipair].i == ibasis)
                    {
                        int ispin = dmi_pairs[ipair].i + icell*geometry->n_cell_atoms;
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[ipair]);
                        if (jspin >= 0)
                        {
                            Energy -= 0.5 * this->dmi_magnitudes[ipair] * this->dmi_normals[ipair].dot(spins[ispin].cross(spins[jspin]));
                        }
                        #ifndef _OPENMP
                        jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[ipair], true);
                        if (jspin >= 0)
                        {
                            Energy += 0.5 * this->dmi_magnitudes[ipair] * this->dmi_normals[ipair].dot(spins[ispin].cross(spins[jspin]));
                        }
                        #endif
                    }
                }
            }

            // DDI
            if (this->idx_ddi >= 0)
            {
                for (unsigned int ipair = 0; ipair < ddi_pairs.size(); ++ipair)
                {
                    if (ddi_pairs[ipair].i == ibasis)
                    {
                        int ispin = ddi_pairs[ipair].i + icell*geometry->n_cell_atoms;
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, ddi_pairs[ipair]);

                        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
                        const scalar mult = 0.5 * mu_s[ispin] * mu_s[jspin]
                            * C::mu_0 * std::pow(C::mu_B, 2) / ( 4*C::Pi * 1e-30 );

                        if (jspin >= 0)
                        {
                            Energy -= mult / std::pow(this->ddi_magnitudes[ipair], 3.0) *
                                (3 * spins[ispin].dot(this->ddi_normals[ipair]) * spins[ispin].dot(this->ddi_normals[ipair]) - spins[ispin].dot(spins[ispin]));

                        }
                        #ifndef _OPENMP
                        jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, dmi_pairs[ipair], true);
                        if (jspin >= 0)
                        {
                            Energy += mult / std::pow(this->ddi_magnitudes[ipair], 3.0) *
                                (3 * spins[ispin].dot(this->ddi_normals[ipair]) * spins[ispin].dot(this->ddi_normals[ipair]) - spins[ispin].dot(spins[ispin]));
                        }
                        #endif
                    }
                }
            }

            // Quadruplets
            if (this->idx_quadruplet >= 0) 
            {
                for (unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad)
                {
                    auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, icell);
                    int ispin = quadruplets[iquad].i + icell*geometry->n_cell_atoms;
                    int jspin = quadruplets[iquad].j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j);
                    int kspin = quadruplets[iquad].k + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k);
                    int lspin = quadruplets[iquad].l + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l);
                    
                    if ( check_atom_type(this->geometry->atom_types[ispin]) && check_atom_type(this->geometry->atom_types[jspin]) &&
                        check_atom_type(this->geometry->atom_types[kspin]) && check_atom_type(this->geometry->atom_types[lspin]) )
                    {
                        Energy -= 0.25*quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * (spins[kspin].dot(spins[lspin]));
                    }

                    #ifndef _OPENMP
                    // TODO: mirrored quadruplet when unique quadruplets are used
                    // jspin = quadruplets[iquad].j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j, true);
                    // kspin = quadruplets[iquad].k + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k, true);
                    // lspin = quadruplets[iquad].l + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l, true);
                    
                    // if ( check_atom_type(this->geometry->atom_types[ispin]) && check_atom_type(this->geometry->atom_types[jspin]) &&
                    //      check_atom_type(this->geometry->atom_types[kspin]) && check_atom_type(this->geometry->atom_types[lspin]) )
                    // {
                    //     Energy -= 0.25*quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * (spins[kspin].dot(spins[lspin]));
                    // }
                    #endif
                }
            }
        }
        return Energy;
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
        // this->Gradient_DDI(spins, gradient);
        // this->Gradient_DDI_direct(spins, gradient);
        this->Gradient_DDI_FFT(spins, gradient);

        // Quadruplets
        this->Gradient_Quadruplet(spins, gradient);
    }

    void Hamiltonian_Heisenberg::Gradient_Zeeman(vectorfield & gradient)
    {
        const int N = geometry->n_cell_atoms;
        auto& mu_s = this->geometry->mu_s;

        #pragma omp parallel for
        for (int icell = 0; icell < geometry->n_cells_total; ++icell)
        {
            for (int ibasis = 0; ibasis < N; ++ibasis)
            {
                int ispin = icell*N + ibasis;
                if (check_atom_type(this->geometry->atom_types[ispin]))
                    gradient[ispin] -= mu_s[ispin] * this->external_field_magnitude * this->external_field_normal;
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
        auto& mu_s = this->geometry->mu_s;
        // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        const scalar mult = C::mu_0 * std::pow(C::mu_B, 2) / ( 4*C::Pi * 1e-30 );
        
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
                                gradient[ispin] -= mu_s[jspin] * skalar_contrib * (3 * ddi_normals[i_pair] * spins[jspin].dot(ddi_normals[i_pair]) - spins[jspin]);
                                gradient[jspin] -= mu_s[ispin] * skalar_contrib * (3 * ddi_normals[i_pair] * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin]);
                            }
                        }
                    }
                }
            }
        }
    }//end Field_DipoleDipole

    void Hamiltonian_Heisenberg::Gradient_DDI_FFT(const vectorfield & spins, vectorfield & gradient)
    {
        //size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        //number of basis atoms (i.e sublattices)
        int B = geometry->n_cell_atoms;

        FFT_spins(spins);
        
        auto& ft_D_matrices = fft_plan_d.cpx_ptr;
        auto& ft_spins = fft_plan_spins.cpx_ptr;

        auto& res_iFFT = fft_plan_rev.real_ptr;
        auto& res_mult = fft_plan_rev.cpx_ptr;

        // auto mapSpins = Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
        //                 (reinterpret_cast<std::complex<scalar>*>(NULL),
        //                 Eigen::Stride<1,Eigen::Dynamic>(1,N));
                    
        // auto mapResult = Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
        //                 (reinterpret_cast<std::complex<scalar>*>(NULL),
        //                 Eigen::Stride<1,Eigen::Dynamic>(1,N));


        for(int i_b1 = 0; i_b1 < B; ++i_b1)
        {
            //should be removed
            std::fill(fft_plan_rev.cpx_ptr.data(), fft_plan_rev.cpx_ptr.data() + 3 * N * geometry->n_cell_atoms, FFT::FFT_cpx_type());
            std::fill(res_iFFT.data(), res_iFFT.data() + 3 * N * geometry->n_cell_atoms, 0.0);
            
            #pragma omp parallel for
            for(int i_b2 = 0; i_b2 < B; ++i_b2)
            {
                // Look up at which position the correct D-matrices are saved
                int b_diff = b_diff_lookup[i_b1 + i_b2 * geometry->n_cell_atoms];

                for(int c = 0; c < Npad[2]; ++c)
                {
                    for(int b = 0; b < Npad[1]; ++b)
                    {
                        for(int a = 0; a < Npad[0]; ++a)
                        {
                            // int idx = a + b * Npad[0] + c * Npad[0] * Npad[1];

                            // auto& D_mat = d_mats_ft[idx + b_diff * N];

                            // //need two maps now one for output, one for spins
                            // new (&mapSpins) Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                            //     (reinterpret_cast<std::complex<scalar>*>(ft_spins.data() + idx + i_b2 * 3 * N),
                            //     Eigen::Stride<1,Eigen::Dynamic>(1,N));

                            // new (&mapResult) Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                            //      (reinterpret_cast<std::complex<scalar>*>(res_mult.data() + idx + i_b1 * 3 * N),
                            //      Eigen::Stride<1,Eigen::Dynamic>(1,N));

                            // mapResult += D_mat * mapSpins;
                            
                            int idx_b2 = i_b2 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c; 
                            int idx_b1 = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c; 
                            int idx_d  = b_diff * d_stride.basis + a * d_stride.a + b * d_stride.b + c * d_stride.c; 
  
                            auto& fs_x = ft_spins[idx_b2                       ];
                            auto& fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
                            auto& fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

                            auto& fD_xx = ft_D_matrices[idx_d                    ];
                            auto& fD_xy = ft_D_matrices[idx_d + 1 * d_stride.comp];
                            auto& fD_xz = ft_D_matrices[idx_d + 2 * d_stride.comp];
                            auto& fD_yy = ft_D_matrices[idx_d + 3 * d_stride.comp];
                            auto& fD_yz = ft_D_matrices[idx_d + 4 * d_stride.comp];
                            auto& fD_zz = ft_D_matrices[idx_d + 5 * d_stride.comp];

                            FFT::addTo(res_mult[idx_b1 + 0 * spin_stride.comp], FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z));
                            FFT::addTo(res_mult[idx_b1 + 1 * spin_stride.comp], FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z));
                            FFT::addTo(res_mult[idx_b1 + 2 * spin_stride.comp], FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z));
                        }
                    }
                }//end iteration over padded lattice cells
            }//end iteration over second sublattice
 
            //Inverse Fourier Transform
            FFT::batch_iFour_3D(fft_plan_rev);

            //Place the gradients at the correct positions and mult with correct mu
            for(int c = 0; c < geometry->n_cells[2]; ++c)
            {
                for(int b = 0; b < geometry->n_cells[1]; ++b)
                {
                    for(int a = 0; a < geometry->n_cells[0]; ++a)
                    {
                        int idx_orig = i_b1 + B * (a + Na * (b + Nb * c));
                        int idx = i_b1 * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c; 
                        gradient[idx_orig][0] -= res_iFFT[idx                       ] / N;   
                        gradient[idx_orig][1] -= res_iFFT[idx + 1 * spin_stride.comp] / N;     
                        gradient[idx_orig][2] -= res_iFFT[idx + 2 * spin_stride.comp] / N;   
                    }
                }
            }  
        }//end iteration sublattice 1
    }

    void Hamiltonian_Heisenberg::Gradient_DDI_direct(const vectorfield & spins, vectorfield & gradient)
    {
        scalar mult = 2 * C::mu_0 * std::pow(C::mu_B, 2) / ( 4*C::Pi * 1e-30 );
        scalar diff, d, d3, d5, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz;
    
        for(int idx1 = 0; idx1 < geometry->nos; idx1++)
        {
            for(int idx2 = 0; idx2 < geometry->nos; idx2++)
            {
                if(idx1 != idx2)
                {
                    auto& m2 = spins[idx2];
                  
                    auto diff = this->geometry->positions[idx2] - this->geometry->positions[idx1];
                    auto d = diff.norm();
                    auto d3 = d * d * d;
                    auto d5 = d * d * d * d * d;
                    scalar Dxx = mult * (3 * diff[0]*diff[0] / d5 - 1/d3);
                    scalar Dxy = mult *  3 * diff[0]*diff[1] / d5;          //same as Dyx
                    scalar Dxz = mult *  3 * diff[0]*diff[2] / d5;          //same as Dzx
                    scalar Dyy = mult * (3 * diff[1]*diff[1] / d5 - 1/d3);
                    scalar Dyz = mult *  3 * diff[1]*diff[2] / d5;          //same as Dzy
                    scalar Dzz = mult * (3 * diff[2]*diff[2] / d5 - 1/d3);

                    auto mu = geometry->mu_s[idx2 % geometry->n_cell_atoms];

                    gradient[idx1][0] -= (Dxx * m2[0] + Dxy * m2[1] + Dxz * m2[2]) * mu;
                    gradient[idx1][1] -= (Dxy * m2[0] + Dyy * m2[1] + Dyz * m2[2]) * mu;                    
                    gradient[idx1][2] -= (Dxz * m2[0] + Dyz * m2[1] + Dzz * m2[2]) * mu;
                }
            }
        }
    }


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
        //		* mu_s[idx_1] * mu_s[idx_2]
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

    void Hamiltonian_Heisenberg::FFT_spins(const vectorfield & spins)
    {
        //size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];
        //bravais vectors
        Vector3 ta = geometry->bravais_vectors[0];
        Vector3 tb = geometry->bravais_vectors[1];
        Vector3 tc = geometry->bravais_vectors[2];
        int B = geometry->n_cell_atoms;

        auto& fft_spin_inputs = fft_plan_spins.real_ptr;

        for(int bi = 0; bi < B; ++bi)
        {   
            //iterate over the **original** system    
            for(int c = 0; c < Nc; ++c)
            {
                for(int b = 0; b < Nb; ++b)
                {
                    for(int a = 0; a < Na; ++a)
                    {
                        // int idx_orig = idx_from_tupel({bi, a, b, c}, {B, Na, Nb, Nc});
                        // #ifdef SPIRIT_USE_FFTW
                        //         int component_stride = 1;
                        //         int idx = idx_from_tupel({0, bi, a, b, c}, {3, B, Npad[0], Npad[1], Npad[2]});
                        // #else
                        //         int component_stride = N;
                        //         int idx = idx_from_tupel({a, b, c, 0, bi}, {Npad[0], Npad[1], Npad[2], 3, B});
                        // #endif
                        int idx_orig = bi + B * (a + Na * (b + Nb * c));
                        int idx = bi * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c; 

                        fft_spin_inputs[idx                        ] = spins[idx_orig][0] * geometry->mu_s[bi];
                        fft_spin_inputs[idx + 1 * spin_stride.comp ] = spins[idx_orig][1] * geometry->mu_s[bi];
                        fft_spin_inputs[idx + 2 * spin_stride.comp ] = spins[idx_orig][2] * geometry->mu_s[bi];                                            
                    }
                }
            }
        }//end iteration over basis
        FFT::batch_Four_3D(fft_plan_spins);
    }

    void Hamiltonian_Heisenberg::FFT_Dipole_Mats(int img_a, int img_b, int img_c)
    {
        //prefactor of ddi interaction
        scalar mult = 2 * C::mu_0 * std::pow(C::mu_B, 2) / ( 4*C::Pi * 1e-30 );

        //size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];
        //bravais vectors
        Vector3 ta = geometry->bravais_vectors[0];
        Vector3 tb = geometry->bravais_vectors[1];
        Vector3 tc = geometry->bravais_vectors[2];
        //number of basis atoms (i.e sublattices)
        int B = geometry->n_cell_atoms;

        auto& fft_dipole_inputs = fft_plan_d.real_ptr;

        int count = -1;
        for(int i_b1 = 0; i_b1 < B; ++i_b1)
        {
            for(int i_b2 = 0; i_b2 < B; ++i_b2)
            {
                if(i_b1 == i_b2 && i_b1 !=0)
                {
                    b_diff_lookup[i_b1 + i_b2 * geometry->n_cell_atoms] = 0;
                    continue;
                }
                count++;
                b_diff_lookup[i_b1 + i_b2 * geometry->n_cell_atoms] = count;

                //iterate over the padded system    
                for(int c = 0; c < Npad[2]; ++c)
                {
                    for(int b = 0; b < Npad[1]; ++b)
                    {
                        for(int a = 0; a < Npad[0]; ++a)
                        {
                            int a_idx = a < Na ? a : a - Npad[0];
                            int b_idx = b < Nb ? b : b - Npad[1];
                            int c_idx = c < Nc ? c : c - Npad[2];
                            scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                            Vector3 diff;
                            //iterate over periodic images

                            for(int a_pb = - img_a + 1; a_pb < img_a; a_pb++)
                            {
                                for(int b_pb = - img_b + 1; b_pb < img_b; b_pb++)
                                {
                                    for(int c_pb = -img_c + 1; c_pb < img_c; c_pb++)
                                    {
                                        diff =    (a_idx + a_pb * Na + geometry->cell_atoms[i_b1][0] - geometry->cell_atoms[i_b2][0]) * ta 
                                                + (b_idx + b_pb * Nb + geometry->cell_atoms[i_b1][1] - geometry->cell_atoms[i_b2][1]) * tb  
                                                + (c_idx + c_pb * Nc + geometry->cell_atoms[i_b1][2] - geometry->cell_atoms[i_b2][2]) * tc;
                                        if(diff.norm() > 1e-10)
                                        {
                                            auto d = diff.norm();
                                            auto d3 = d * d * d;
                                            auto d5 = d * d * d * d * d;
                                            Dxx += mult * (3 * diff[0]*diff[0] / d5 - 1/d3);
                                            Dxy += mult *  3 * diff[0]*diff[1] / d5;          //same as Dyx
                                            Dxz += mult *  3 * diff[0]*diff[2] / d5;          //same as Dzx
                                            Dyy += mult * (3 * diff[1]*diff[1] / d5 - 1/d3);
                                            Dyz += mult *  3 * diff[1]*diff[2] / d5;          //same as Dzy
                                            Dzz += mult * (3 * diff[2]*diff[2] / d5 - 1/d3);
                                        } 
                                    }
                                }
                            }

                            int idx = count * d_stride.basis + a * d_stride.a + b * d_stride.b + c * d_stride.c;

                            fft_dipole_inputs[idx                    ] = Dxx;
                            fft_dipole_inputs[idx + 1 * d_stride.comp] = Dxy;
                            fft_dipole_inputs[idx + 2 * d_stride.comp] = Dxz;
                            fft_dipole_inputs[idx + 3 * d_stride.comp] = Dyy;
                            fft_dipole_inputs[idx + 4 * d_stride.comp] = Dyz;
                            fft_dipole_inputs[idx + 5 * d_stride.comp] = Dzz;

                            // Vector3 diff =    (a_idx + geometry->cell_atoms[i_b1][0] - geometry->cell_atoms[i_b2][0]) * ta 
                            //                 + (b_idx + geometry->cell_atoms[i_b1][1] - geometry->cell_atoms[i_b2][1]) * tb  
                            //                 + (c_idx + geometry->cell_atoms[i_b1][2] - geometry->cell_atoms[i_b2][2]) * tc; 

                            // if(!(a==0 && b==0 && c==0 && i_b1 == i_b2))
                            // {
                            //     auto d = diff.norm();
                            //     auto d3 = d * d * d;
                            //     auto d5 = d * d * d * d * d;
                            //     scalar Dxx = mult * (3 * diff[0]*diff[0] / d5 - 1/d3);
                            //     scalar Dxy = mult *  3 * diff[0]*diff[1] / d5;          //same as Dyx
                            //     scalar Dxz = mult *  3 * diff[0]*diff[2] / d5;          //same as Dzx
                            //     scalar Dyy = mult * (3 * diff[1]*diff[1] / d5 - 1/d3);
                            //     scalar Dyz = mult *  3 * diff[1]*diff[2] / d5;          //same as Dzy
                            //     scalar Dzz = mult * (3 * diff[2]*diff[2] / d5 - 1/d3);

                            //     int idx = count * d_stride.basis + a * d_stride.a + b * d_stride.b + c * d_stride.c;

                            //     fft_dipole_inputs[idx                    ] = Dxx;
                            //     fft_dipole_inputs[idx + 1 * d_stride.comp] = Dxy;
                            //     fft_dipole_inputs[idx + 2 * d_stride.comp] = Dxz;
                            //     fft_dipole_inputs[idx + 3 * d_stride.comp] = Dyy;
                            //     fft_dipole_inputs[idx + 4 * d_stride.comp] = Dyz;
                            //     fft_dipole_inputs[idx + 5 * d_stride.comp] = Dzz;
                            // }
                        }
                    }
                }
            }
        }
        FFT::batch_Four_3D(fft_plan_d);
    }

    void Hamiltonian_Heisenberg::Prepare_DDI()
    {
        Npad.resize(3);
        Npad[0] = (geometry->n_cells[0] > 1) ? 2 * geometry->n_cells[0] : 1;
        Npad[1] = (geometry->n_cells[1] > 1) ? 2 * geometry->n_cells[1] : 1;
        Npad[2] = (geometry->n_cells[2] > 1) ? 2 * geometry->n_cells[2] : 1;
        N = Npad[0] * Npad[1] * Npad[2];

        b_diff_lookup.resize(geometry->n_cell_atoms * geometry->n_cell_atoms);


        //we dont need to transform over length 1 dims
        std::vector<int> fft_dims;
        for(int i = 2; i >= 0; i--) //notice that reverse order is important!
        {
            if(Npad[i] > 1)
                fft_dims.push_back(Npad[i]);
        }

        //Count how many distinct inter-lattice contributions we need to store
        symmetry_count = 0;
        for(int i = 0; i < geometry->n_cell_atoms; i++)
        {   
            for(int j = 0; j < geometry->n_cell_atoms; j++)
            {
                if(i != 0 && i==j) continue;
                symmetry_count++;
            }
        }

        //Create fft plans.
        fft_plan_d.dims     = fft_dims;
        fft_plan_d.inverse  = false;
        fft_plan_d.howmany  = 6 * symmetry_count;
        fft_plan_d.real_ptr = field<FFT::FFT_real_type>(symmetry_count * 6 * N);
        fft_plan_d.cpx_ptr  = field<FFT::FFT_cpx_type>(symmetry_count * 6 * N);
        fft_plan_d.CreateConfiguration();

        fft_plan_spins.dims     = fft_dims;
        fft_plan_spins.inverse  = false;
        fft_plan_spins.howmany  = 3 * geometry->n_cell_atoms;
        fft_plan_spins.real_ptr = field<FFT::FFT_real_type>(3 * N * geometry->n_cell_atoms);
        fft_plan_spins.cpx_ptr  = field<FFT::FFT_cpx_type>(3 * N * geometry->n_cell_atoms);
        fft_plan_spins.CreateConfiguration();


        fft_plan_rev.dims     = fft_dims;
        fft_plan_rev.inverse  = true;
        fft_plan_rev.howmany  = 3 * geometry->n_cell_atoms;
        fft_plan_rev.cpx_ptr  = field<FFT::FFT_cpx_type>(3 * N * geometry->n_cell_atoms);
        fft_plan_rev.real_ptr = field<FFT::FFT_real_type>(3 * N * geometry->n_cell_atoms);
        fft_plan_rev.CreateConfiguration();

        #ifdef SPIRIT_USE_FFTW
            field<int*> temp_s = {&spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c};
            field<int*> temp_d = {&d_stride.comp, &d_stride.basis, &d_stride.a, &d_stride.b, &d_stride.c};;
            FFT::get_strides(temp_s, {3, this->geometry->n_cell_atoms, Npad[0], Npad[1], Npad[2]});
            FFT::get_strides(temp_d, {6, symmetry_count, Npad[0], Npad[1], Npad[2]});
        #else
            field<int*> temp_s = {&spin_stride.a, &spin_stride.b, &spin_stride.c, &spin_stride.comp, &spin_stride.basis};
            field<int*> temp_d = {&d_stride.a, &d_stride.b, &d_stride.c, &d_stride.comp, &d_stride.basis};;
            FFT::get_strides(temp_s, {Npad[0], Npad[1], Npad[2], 3, this->geometry->n_cell_atoms});
            FFT::get_strides(temp_d, {Npad[0], Npad[1], Npad[2], 6, symmetry_count});    
        #endif

        //perform FFT of dipole matrices
        int img_a = boundary_conditions[0] == 0 ? 1 : 10;
        int img_b = boundary_conditions[1] == 0 ? 1 : 10;
        int img_c = boundary_conditions[2] == 0 ? 1 : 10;
        FFT_Dipole_Mats(img_a, img_b, img_c);

        d_mats_ft = field<Matrix3c>(N * symmetry_count);

        //Write out ft dipole matrices
        for(int b_diff = 0; b_diff < symmetry_count; ++b_diff)
        {
            for(int c = 0; c < Npad[2]; ++c)
            {
                for(int b = 0; b < Npad[1]; ++b)
                {
                    for(int a = 0; a < Npad[0]; ++a)
                    {
                        int idx = b_diff * d_stride.basis + a * d_stride.a + b * d_stride.b + c * d_stride.c;
                        auto fD_xx = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx                    ]);
                        auto fD_xy = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + d_stride.comp * 1]);
                        auto fD_xz = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + d_stride.comp * 2]);
                        auto fD_yy = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + d_stride.comp * 3]);
                        auto fD_yz = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + d_stride.comp * 4]);
                        auto fD_zz = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + d_stride.comp * 5]);
                        d_mats_ft[b_diff + symmetry_count * (a + Npad[0] * (b + Npad[1] * c))] <<  *fD_xx, *fD_xy, *fD_xz,
                                                                                                                               *fD_xy, *fD_yy, *fD_yz,
                                                                                                                              *fD_xz, *fD_yz, *fD_zz;
                    }
                }
            }
        }
    }

    // Hamiltonian name as string
    static const std::string name = "Heisenberg";
    const std::string& Hamiltonian_Heisenberg::Name() { return name; }
}

#endif