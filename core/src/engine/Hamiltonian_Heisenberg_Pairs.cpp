#ifndef USE_CUDA

#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include <Spirit_Defines.h>
#include <engine/Hamiltonian_Heisenberg_Pairs.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>


using std::vector;
using std::function;

using namespace Data;
using namespace Utility;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;

namespace Engine
{
    Hamiltonian_Heisenberg_Pairs::Hamiltonian_Heisenberg_Pairs(
        scalarfield mu_s,
        intfield external_field_indices, scalarfield external_field_magnitudes, vectorfield external_field_normals,
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
        external_field_indices(external_field_indices), external_field_magnitudes(external_field_magnitudes), external_field_normals(external_field_normals),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_pairs(exchange_pairs), exchange_magnitudes(exchange_magnitudes),
        dmi_pairs(dmi_pairs), dmi_magnitudes(dmi_magnitudes), dmi_normals(dmi_normals),
        quadruplets(quadruplets), quadruplet_magnitudes(quadruplet_magnitudes)
    {
        // Renormalize the external field from Tesla to whatever
        for (unsigned int i = 0; i < external_field_magnitudes.size(); ++i)
        {
            this->external_field_magnitudes[i] = this->external_field_magnitudes[i] * Constants::mu_B * mu_s[i];
        }

        // Generate DDI pairs, magnitudes, normals
        this->ddi_pairs = Engine::Neighbours::Get_Pairs_in_Radius(*this->geometry, ddi_radius);
        scalar magnitude;
        Vector3 normal;
        for (unsigned int i = 0; i<ddi_pairs.size(); ++i)
        {
            Engine::Neighbours::DDI_from_Pair(*this->geometry, { ddi_pairs[i].i, ddi_pairs[i].j, ddi_pairs[i].translations }, magnitude, normal);
            this->ddi_magnitudes.push_back(magnitude);
            this->ddi_normals.push_back(normal);
        }

        // Generate DDI macrocell matrices etc.
        this->Prepare_MacroCells();

        // Set the array of energy contributions
        this->Update_Energy_Contributions();
    }


    void Hamiltonian_Heisenberg_Pairs::Update_From_Geometry()
    {
        // TODO: data needs to be scaled and ordered correctly
        // TODO: there should be a basic set of info given through the constructor,
        //       i.e. per basis cell, which can then be extrapolated. This needs to
        //       redesigned.

        this->anisotropy_indices.resize(this->geometry->nos);
        this->anisotropy_magnitudes.resize(this->geometry->nos);
        this->anisotropy_normals.resize(this->geometry->nos);

        this->external_field_indices.resize(this->geometry->nos);
        this->external_field_magnitudes.resize(this->geometry->nos);
        this->external_field_normals.resize(this->geometry->nos);
    }


    void Hamiltonian_Heisenberg_Pairs::Update_Energy_Contributions()
    {
        this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

        // External field
        if (this->external_field_indices.size() > 0)
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
        // Dipole-Dipole (macrocell approximation)
        if (this->n_mc_atoms > 0)
        {
            this->energy_contributions_per_spin.push_back({"DDI-MC", scalarfield(0) });
            this->idx_ddi_mc = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_ddi_mc = -1;
        // Quadruplets
        if (this->quadruplets.size() > 0)
        {
            this->energy_contributions_per_spin.push_back({"Quadruplets", scalarfield(0) });
            this->idx_quadruplet = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_quadruplet = -1;
    }

    void Hamiltonian_Heisenberg_Pairs::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
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
        // DDI
        if (this->idx_ddi >=0 )        E_DDI(spins, contributions[idx_ddi].second);
        // DDI (macrocells)
        if (this->idx_ddi_mc >= 0 )    E_DDI_MacroCells(spins, contributions[idx_ddi_mc].second);
        // Quadruplets
        if (this->idx_quadruplet >=0 ) E_Quadruplet(spins, contributions[idx_quadruplet].second);
    }

    void Hamiltonian_Heisenberg_Pairs::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        #pragma omp parallel for //print at start
        for (unsigned int i = 0; i < this->external_field_indices.size(); ++i)
        {
            int ispin = external_field_indices[i];
            if ( check_atom_type(this->geometry->atom_types[ispin]) )
                #pragma omp atomic
                Energy[ispin] -= this->external_field_magnitudes[i] * this->external_field_normals[i].dot(spins[ispin]);
        }
    }

    void Hamiltonian_Heisenberg_Pairs::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
    {
        #pragma omp parallel for
        for (unsigned int i = 0; i < this->anisotropy_indices.size(); ++i)
        {
            int ispin = anisotropy_indices[i];
            if ( check_atom_type(this->geometry->atom_types[ispin]) )
                #pragma omp atomic
                Energy[ispin] -= this->anisotropy_magnitudes[i] * std::pow(anisotropy_normals[i].dot(spins[ispin]), 2.0);
        }
    }

    void Hamiltonian_Heisenberg_Pairs::E_Exchange(const vectorfield & spins, scalarfield & Energy)
    {
        const int Na = geometry->n_cells[0];
        const int Nb = geometry->n_cells[1];
        const int Nc = geometry->n_cells[2];
        //print at start
        #pragma omp parallel for collapse(3)
        for (int da = 0; da < Na; ++da)
        {
            for (int db = 0; db < Nb; ++db)
            {
                for (int dc = 0; dc < Nc; ++dc)
                {
                    for (unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        int ispin = exchange_pairs[i_pair].i+ Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, exchange_pairs[i_pair]);
                        if (jspin >= 0)
                        {
                            Energy[ispin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin]);
                            #ifndef _OPENMP
                            Energy[jspin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin]);
                            #endif
                        }

                        // To parallelize with OpenMP we avoid atomics by not adding to two different spins in one thread.
                        //		instead, we need to also add the inverse pair to each spin, which makes it similar to the
                        //		neighbours implementation (in terms of the number of pairs)
                        #ifdef _OPENMP
                        int jspin2 = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, exchange_pairs[i_pair], true);
                        if (jspin2 >= 0)
                        {
                            Energy[ispin] -= 0.5 * exchange_magnitudes[i_pair] * spins[ispin].dot(spins[jspin2]);
                        }
                        #endif
                    }
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Pairs::E_DMI(const vectorfield & spins, scalarfield & Energy)
    {
        const int Na = geometry->n_cells[0];
        const int Nb = geometry->n_cells[1];
        const int Nc = geometry->n_cells[2];
        //print at start
        #pragma omp parallel for collapse(3)
        for (int da = 0; da < Na; ++da)
        {
            for (int db = 0; db < Nb; ++db)
            {
                for (int dc = 0; dc < Nc; ++dc)
                {
                    for (unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        int ispin = dmi_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, dmi_pairs[i_pair]);
                        if (jspin >= 0)
                        {
                            Energy[ispin] -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin]));
                            #ifndef _OPENMP
                            Energy[jspin] -= 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin]));
                            #endif
                        }

                        // To parallelize with OpenMP we avoid atomics by not adding to two different spins in one thread.
                        //		instead, we need to also add the inverse pair to each spin, which makes it similar to the
                        //		neighbours implementation (in terms of the number of pairs)
                        #ifdef _OPENMP
                        int jspin2 = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, dmi_pairs[i_pair], true);
                        if (jspin2 >= 0)
                        {
                            Energy[ispin] += 0.5 * dmi_magnitudes[i_pair] * dmi_normals[i_pair].dot(spins[ispin].cross(spins[jspin2]));
                        }
                        #endif
                    }
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Pairs::E_DDI(const vectorfield & spins, scalarfield & Energy)
    {
        //scalar mult = -Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
        // factor 0.5 in mult because we have each pair twice in our list, 0.5 below in the energy because we add the term twice
        scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        scalar result = 0.0;
        //Print after set the radius on spirit
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
                            int ispin = ddi_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                            int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, ddi_pairs[i_pair]);
                            if (jspin >= 0)
                            {
                                Energy[ispin] -= 0.5*mult / std::pow(ddi_magnitudes[i_pair], 3.0) * mu_s[0] * mu_s[0] *
                                    (3 * spins[ispin].dot(ddi_normals[i_pair]) * spins[jspin].dot(ddi_normals[i_pair]) - spins[ispin].dot(spins[jspin]));
                                Energy[jspin] -= 0.5*mult / std::pow(ddi_magnitudes[i_pair], 3.0) * mu_s[0] * mu_s[0] *
                                    (3 * spins[ispin].dot(ddi_normals[i_pair]) * spins[jspin].dot(ddi_normals[i_pair]) - spins[ispin].dot(spins[jspin]));
                            }
                        }
                    }
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Pairs::Prepare_MacroCells()
    {
        // --- Hard-coded macro-cell structure
        // Number of basis cells in a macrocell (along a b c)
        intfield n_cells_in_mc = intfield({ 2,2,2 });

        // Number of atoms in the macro cell (e.g 8 is a cubic macro cell)
        this->n_mc_atoms = this->geometry->n_spins_basic_domain * n_cells_in_mc[0] * n_cells_in_mc[1] * n_cells_in_mc[2];

        const int na = geometry->n_cells[0];
        const int nb = geometry->n_cells[1];
        const int nc = geometry->n_cells[2];

        // Number of macro cells
        this->n_cells_macro = intfield({ na/n_cells_in_mc[0], nb/n_cells_in_mc[1], nc/n_cells_in_mc[2] });
        this->n_mc_total = n_cells_macro[0] * n_cells_macro[1] * n_cells_macro[2];

        const int na_mc = n_cells_macro[0];
        const int nb_mc = n_cells_macro[1];
        const int nc_mc = n_cells_macro[2];

        // --- Initialization
        this->atom_id_mc   = std::vector<intfield>    ( n_mc_total, intfield(n_mc_atoms, -1) );
        this->xyz_atoms_mc = std::vector<vectorfield> ( n_mc_total, vectorfield(n_mc_atoms, Vector3::Zero()) );

        // --- Determine the indices and positions of atoms in the macrocells
        // Loop over macro cells
        for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
        {
            int atom_mc = 0;
            // Calculate the atom index of the first atom in the macrocell
            int mc_idx_c = p_mc / (na_mc*nb_mc);
            int mc_idx_b = (p_mc - mc_idx_c * na_mc*nb_mc) / na_mc;
            int mc_idx_a = p_mc - mc_idx_c * na_mc*nb_mc - mc_idx_b * na_mc;

            int mc_first_index = mc_idx_a * n_cells_in_mc[0]
                                + mc_idx_b * na_mc*n_cells_in_mc[0]*n_cells_in_mc[1]
                                + mc_idx_c * na_mc*nb_mc*n_cells_in_mc[0]*n_cells_in_mc[1]*n_cells_in_mc[2];

            // Loop over spins in the macrocell
            for (int idx_c = 0; idx_c < n_cells_in_mc[0]; ++idx_c)
            {
                for (int idx_b = 0; idx_b < n_cells_in_mc[1]; ++idx_b)
                {
                    for (int idx_a = 0; idx_a < n_cells_in_mc[2]; ++idx_a)
                    {
                        // Calculate the index
                        int atom_id = mc_first_index + idx_a + idx_b*na + idx_c*na*nb;
                        // Set the index
                        atom_id_mc[p_mc][atom_mc] = atom_id;
                        // Set the position
                        xyz_atoms_mc[p_mc][atom_mc] = geometry->spin_pos[atom_id];
                        // Increment the atom index inside the macrocell
                        ++atom_mc;
                    }
                }
            }
        }//end loop over macro-cells

        //// --- NOTE: we do not need the intra-matrices if we do the intra-cell calculations precisely!
        //// --- Calculate the dipole-dipole matrices inside the macrocells (intra)
        //this->D = std::vector<Matrix3>(n_mc_total, Matrix3::Zero());
        //Matrix3 Dij;
        //// Loop over macro cells
        //for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
        //{
        //    // Loop over atom i in macrocell (pi)
        //    for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
        //    {
        //        int  id_i = atom_id_mc[p_mc][atom_i];
        //        // Loop over atom j in macrocell (pj)
        //        for (int atom_j = 0; atom_j < n_mc_atoms; ++atom_j)
        //        {
        //            // Exclude self-interactions
        //            if (atom_i != atom_j)
        //            {
        //                int  id_j = atom_id_mc[p_mc][atom_j];

        //                // Relative distance between atom_i and atom_j in the mc
        //                Vector3 r_vec = xyz_atoms_mc[p_mc][atom_j] - xyz_atoms_mc[p_mc][atom_i];
        //                scalar  r = r_vec.norm();

        //                scalar x = r_vec[0];
        //                scalar y = r_vec[1];
        //                scalar z = r_vec[2];

        //                // Get dipole-dipole matrix for the pair of atoms in the macro-cell
        //                Dij << (3 * x*x - r * r), (3 * x*y), (3 * x*z),
        //                       (3 * x*y), (3 * y*y - r * r), (3 * y*z),
        //                       (3 * x*z), (3 * y*z), (3 * z*z - r * r);

        //                scalar term = Constants::mu_B / (4 * M_PI*std::pow(r, 5));

        //                // Add it to the macrocell matrix
        //                this->D[p_mc] += term * Dij;
        //            }
        //        }//End loop over atom_j
        //    }//End loop over atom_i
        //}//End loop over macro cell

        // --- Calculate the dipole-dipole matrices between macrocell pairs (inter)
        this->D_inter = std::vector<std::vector<Matrix3>>(n_mc_total, vector<Matrix3>(n_mc_total, Matrix3::Zero()));
        Matrix3 D_tmp;
        scalar mult = 0.0536814951168;
        // Loop over q macro cell
        for (unsigned int q_mc = 0; q_mc < n_mc_total; ++q_mc)
        {
            // Loop over p macro cell
            for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
            {
                // Exclude self-interactions
                if (q_mc != p_mc)
                {
                    // Loop over atom_j in q_mc -> qj
                    for (int atom_j = 0; atom_j < n_mc_atoms; ++atom_j)
                    {
                        // Loop over atom_i in p_mc -> pi
                        for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
                        {
                            Vector3 r_vec = xyz_atoms_mc[p_mc][atom_i] - xyz_atoms_mc[q_mc][atom_j];
                            scalar  r = r_vec.norm();

                            scalar x = r_vec[0];
                            scalar y = r_vec[1];
                            scalar z = r_vec[2];

                            // TODO: check parameters
                            //scalar term = Constants::mu_B / (4 * M_PI*std::pow(r, 5));
                            scalar term = 0.5*mult / std::pow(r, 5.0) * mu_s[0] * mu_s[0];

                            // Get Effective dipole matrix
                            D_tmp << (3*x*x - r*r), (3*x*y), (3*x*z),
                                     (3*x*y), (3*y*y - r*r), (3*y*z),
                                     (3*x*z), (3*y*z), (3*z*z - r*r);

                            // Add it to the inter dipole dipole matrix
                            this->D_inter[q_mc][p_mc] += term / (n_mc_atoms*n_mc_atoms) * D_tmp;
                        }
                    }
                }
            }//End loop over macro-cells p
        }//End loop over macro-cells q


    }//End Prepare_MacroCells

    void Hamiltonian_Heisenberg_Pairs::Update_MacroSpins(const vectorfield & spins)
    {
        // Get Macrospins
        Vector3 macrospin;
        this->macrospins = vectorfield(n_mc_total, Vector3::Zero());

        // Loop over macro cells
        for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
        {
            macrospin.setZero();
            // Loop over atoms in the mc
            for (int atom_mc = 0; atom_mc < n_mc_atoms; ++atom_mc)
            {
                // Moment contribution of each atom in the mc
                macrospin += mu_s[0] * spins[atom_id_mc[p_mc][atom_mc]];
            }
            // Total moment in the macro cell
            macrospins[p_mc] = macrospin;
        }
    }

    void Hamiltonian_Heisenberg_Pairs::E_DDI_MacroCells(const vectorfield & spins, scalarfield & Energy)
    {
        this->Update_MacroSpins(spins);

        // --- Energy inside mc (intra)
        E_intra = 0.0;
        scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        Matrix3 D_tmp;
        // Loop over macro cells
        for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
        {
            for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
            {
                int  id_i = atom_id_mc[p_mc][atom_i];
                for (int atom_j = 0; atom_j < n_mc_atoms; ++atom_j)
                {
                    // Exclude self-interactions
                    if (atom_i != atom_j)
                    {
                        int  id_j = atom_id_mc[p_mc][atom_j];

                        // Relative distances between atom_i and atom_j in the mc
                        Vector3 r_vec = xyz_atoms_mc[p_mc][atom_j] - xyz_atoms_mc[p_mc][atom_i];
                        scalar  r = r_vec.norm();

                        scalar x = r_vec[0];
                        scalar y = r_vec[1];
                        scalar z = r_vec[2];

                        //scalar term = mu_s[0] * mu_s[0] * Constants::mu_B / (4 * M_PI*std::pow(r, 5));
                        scalar term = mult / std::pow(r, 5.0) * mu_s[0] * mu_s[0];

                        //Get dipole-dipole matrix for the atoms in the macro-cell
                        D_tmp << (3*x*x - r*r), (3*x*y), (3*x*z),
                                 (3*x*y), (3*y*y - r*r), (3*y*z),
                                 (3*x*z), (3*y*z), (3*z*z - r*r);

                        //Sum of the energy contributions
                        E_intra -= term * spins[id_i].dot(D_tmp * spins[id_j]);
                        Energy[id_i] -= term * spins[id_i].dot(D_tmp * spins[id_j]);

                        // // Alternative implementation:
                        // Vector3 ipos = geometry->spin_pos[id_i];
                        // Vector3 jpos = geometry->spin_pos[id_j];

                        // // Calculate positions and difference vector
                        // Vector3 vector_ij = jpos - ipos;

                        // // Length of difference vector
                        // scalar magnitude = vector_ij.norm();
                        // Vector3 normal = vector_ij.normalized();

                        // E_intra += mult / std::pow(magnitude, 3.0) * mu_s[0] * mu_s[0] *
                            //   (3 * spins[id_i].dot(normal) * spins[id_j].dot(normal) - spins[id_i].dot(spins[id_j]));
                    }
                }
            }
        }//End loop over macro cell

        // --- Energy contribution between macro-cells (inter)
        scalar E_dip_mc = 0.0;
        // Loop over q macro cell
        for (unsigned int q_mc = 0; q_mc < n_mc_total; ++q_mc)
        {
            // Loop over p macro cell
            for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
            {
                // Exclude self-interactions
                if (q_mc != p_mc)
                {
                    E_dip_mc -= 0.5 * macrospins[q_mc].dot(D_inter[q_mc][p_mc]*macrospins[p_mc]);
                    for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
                    {
                        int  id_i = atom_id_mc[p_mc][atom_i];
                        Energy[id_i] -= 0.5 * macrospins[q_mc].dot(D_inter[q_mc][p_mc]*macrospins[p_mc]);
                    }
                }
            }//end loop over macro-cells p
        }//end loop over macro-cells q

        //////// Only for debugging ///////
        // Print total dipole dipole energy
        double Etotal = E_dip_mc + E_intra;
        std::cout << std::endl;
        std::cout << " --Get energy-- " << std::endl;
        std::cout << E_dip_mc/geometry->nos << " + " << E_intra/geometry->nos << " = " << Etotal/geometry->nos << std::endl;
        std::cout << std::endl;

        // Call gradient calculation so those values are also printed
        vectorfield tmp(geometry->nos, { 0,0,0 });
        Gradient_DDI_MacroCells(spins, tmp);
        ///////////////////////////////////
    }// end Dipole-Dipole Energy


    void Hamiltonian_Heisenberg_Pairs::Gradient_DDI_MacroCells(const vectorfield& spins, vectorfield & gradient)
    {
        this->Update_MacroSpins(spins);

        // Contribution inside the macro-cell  (for squared mc should be zero)
        this->grad_E_in = vectorfield(geometry->nos, Vector3::Zero());
        //Vector3 grad_E_in{0,0,0};
        Matrix3 D_tmp;
        scalar mult = 0.0536814951168;
        // Loop over macro cells
        for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
        {
            for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
            {
                int  id_i = atom_id_mc[p_mc][atom_i];
                for (int atom_j = 0; atom_j < n_mc_atoms; ++atom_j)
                {
                    // Exclude self-interactions
                    if (atom_i != atom_j)
                    {
                        int  id_j = atom_id_mc[p_mc][atom_j];

                        // Relative distances between atom_i and atom_j in the mc
                        Vector3 r_vec = xyz_atoms_mc[p_mc][atom_j] - xyz_atoms_mc[p_mc][atom_i];
                        scalar  r = r_vec.norm();

                        scalar x = r_vec[0];
                        scalar y = r_vec[1];
                        scalar z = r_vec[2];

                        //scalar term = mu_s[0] * mu_s[0] * Constants::mu_B / (4 * M_PI*std::pow(r, 5));
                        scalar term = mult / std::pow(r, 5.0) * mu_s[0] * mu_s[0];

                        //Get dipole-dipole matrix for the atoms in the macro-cell
                        D_tmp << (3*x*x - r*r), (3*x*y), (3*x*z),
                                 (3*x*y), (3*y*y - r*r), (3*y*z),
                                 (3*x*z), (3*y*z), (3*z*z - r*r);

                        //Gradient
                        grad_E_in[id_i] += term * D_tmp * spins[id_j];
                        gradient[id_i] += term * D_tmp * spins[id_j];
                    }
                }//end loop over atom_i
            }//end loop over atom_i
        }//end loop over macro cell

        //Vector3 grad_E_mc{0,0,0};
        this-> grad_E_mc = vectorfield(geometry->nos, Vector3::Zero());

        // Loop over q macro cell
        for (unsigned int q_mc = 0; q_mc < n_mc_total; ++q_mc)
        {
            // Loop over p macro cell
            for (unsigned int p_mc = 0; p_mc < n_mc_total; ++p_mc)
            {
                // Exclude self-interactions
                if (q_mc != p_mc)
                {
                    for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
                    {
                        int  id_i = atom_id_mc[q_mc][atom_i];
                        grad_E_mc[id_i] += 0.5 * D_inter[q_mc][p_mc]*macrospins[p_mc];
                        gradient[id_i] += 0.5 * D_inter[q_mc][p_mc]*macrospins[p_mc];
                    }
                }
            }//end loop over macro-cells p
        }//end loop over macro-cells q

        //Vector3 grad_E;
        this-> grad_E = vectorfield(geometry->nos, Vector3::Zero());

        for (unsigned int q_mc = 0; q_mc < n_mc_total; ++q_mc)
        {
            for (int atom_i = 0; atom_i < n_mc_atoms; ++atom_i)
            {
                int  id_i = atom_id_mc[q_mc][atom_i];
                grad_E[id_i] = -2*grad_E_mc[id_i] - grad_E_in[id_i];
            }
        }

        //////// Only for debugging ///////
        // Print macrocell gradient
        std::cout << std::endl;
        std::cout << " --Get gradient-- "<< std::endl;
        std::cerr << grad_E_in[0].transpose() << std::endl;
        std::cerr << grad_E_mc[0].transpose() << std::endl;
        std::cerr << grad_E[0].transpose() << std::endl;
        // Print old gradient
        std::cout << std::endl;
        Vectormath::fill(grad_E, {0,0,0});
        this->Gradient_DDI(spins, grad_E);
        std::cout << " --Get gradient GIDEON-- "<< std::endl;
        std::cerr << grad_E[0].transpose() << std::endl;
        ///////////////////////////////////
    }


    void Hamiltonian_Heisenberg_Pairs::E_Quadruplet(const vectorfield & spins, scalarfield & Energy)
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
                        int ispin = quadruplets[iquad].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                        int jspin = quadruplets[iquad].j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_j);
                        int kspin = quadruplets[iquad].k + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_k);
                        int lspin = quadruplets[iquad].l + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_l);

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



    void Hamiltonian_Heisenberg_Pairs::Gradient(const vectorfield & spins, vectorfield & gradient)
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
        // DDI
        this->Gradient_DDI(spins, gradient);
        // DDI (macrocells)
        this->Gradient_DDI_MacroCells(spins, gradient);

        // Quadruplets
        this->Gradient_Quadruplet(spins, gradient);
    }

    void Hamiltonian_Heisenberg_Pairs::Gradient_Zeeman(vectorfield & gradient)
    {
        #pragma omp parallel for
        //Print during the running
        for (unsigned int i = 0; i < this->external_field_indices.size(); ++i)
        {
            int ispin = external_field_indices[i];
            if ( check_atom_type(this->geometry->atom_types[ispin]) )
                #pragma omp critical
                gradient[ispin] -= this->external_field_magnitudes[i] * this->external_field_normals[i];
        }
    }

    void Hamiltonian_Heisenberg_Pairs::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for //Print during the running
        for (unsigned int i = 0; i < this->anisotropy_indices.size(); ++i)
        {
            int ispin = anisotropy_indices[i];
            if ( check_atom_type(this->geometry->atom_types[ispin]) )
                #pragma omp critical
                gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[i] * this->anisotropy_normals[i] * anisotropy_normals[i].dot(spins[ispin]);
        }
    }

    void Hamiltonian_Heisenberg_Pairs::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        const int Na = geometry->n_cells[0];
        const int Nb = geometry->n_cells[1];
        const int Nc = geometry->n_cells[2];

        #pragma omp parallel for collapse(3)
        for (int da = 0; da < Na; ++da)
        {
            for (int db = 0; db < Nb; ++db)
            {
                for (int dc = 0; dc < Nc; ++dc)
                {
                    std::array<int, 3> translations = { da, db, dc };
                    for (unsigned int i_pair = 0; i_pair < exchange_pairs.size(); ++i_pair)
                    {
                        int ispin = exchange_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, exchange_pairs[i_pair]);
                        if (jspin >= 0)
                        {
                            gradient[ispin] -= exchange_magnitudes[i_pair] * spins[jspin];
                            #ifndef _OPENMP
                            gradient[jspin] -= exchange_magnitudes[i_pair] * spins[ispin];
                            #endif
                        }

                        // To parallelize with OpenMP we avoid atomics by not adding to two different spins in one thread.
                        //		instead, we need to also add the inverse pair to each spin, which makes it similar to the
                        //		neighbours implementation (in terms of the number of pairs)
                        #ifdef _OPENMP
                        int jspin2 = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, exchange_pairs[i_pair], true);
                        if (jspin2 >= 0)
                        {
                            gradient[ispin] -= exchange_magnitudes[i_pair] * spins[jspin2];
                        }
                        #endif
                    }
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Pairs::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        const int Na = geometry->n_cells[0];
        const int Nb = geometry->n_cells[1];
        const int Nc = geometry->n_cells[2];
        //print during running
        #pragma omp parallel for collapse(3)
        for (int da = 0; da < Na; ++da)
        {
            for (int db = 0; db < Nb; ++db)
            {
                for (int dc = 0; dc < Nc; ++dc)
                {
                    std::array<int, 3 > translations = { da, db, dc };
                    for (unsigned int i_pair = 0; i_pair < dmi_pairs.size(); ++i_pair)
                    {
                        int ispin = dmi_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                        int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, dmi_pairs[i_pair]);
                        if (jspin >= 0)
                        {
                            gradient[ispin] -= dmi_magnitudes[i_pair] * spins[jspin].cross(dmi_normals[i_pair]);
                            #ifndef _OPENMP
                            gradient[jspin] += dmi_magnitudes[i_pair] * spins[ispin].cross(dmi_normals[i_pair]);
                            #endif
                        }

                        // To parallelize with OpenMP we avoid atomics by not adding to two different spins in one thread.
                        //		instead, we need to also add the inverse pair to each spin, which makes it similar to the
                        //		neighbours implementation (in terms of the number of pairs)
                        #ifdef _OPENMP
                        int jspin2 = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, dmi_pairs[i_pair], true);
                        if (jspin2 >= 0)
                        {
                            gradient[ispin] += dmi_magnitudes[i_pair] * spins[jspin2].cross(dmi_normals[i_pair]);
                        }
                        #endif
                    }
                }
            }
        }
    }

    void Hamiltonian_Heisenberg_Pairs::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
    {
        //scalar mult = Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
        scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
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

                            int ispin = ddi_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                            int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_spins_basic_domain, geometry->atom_types, ddi_pairs[i_pair]);
                            if (jspin >= 0)
                            {
                                gradient[ispin] -= skalar_contrib * (3 * ddi_normals[i_pair] * spins[jspin].dot(ddi_normals[i_pair]) - spins[jspin]);
                                gradient[jspin] -= skalar_contrib * (3 * ddi_normals[i_pair] * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin]);
                            }
                        }
                    }
                }
            }
        }
    }//end Field_DipoleDipole


    void Hamiltonian_Heisenberg_Pairs::Gradient_Quadruplet(const vectorfield & spins, vectorfield & gradient)
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
                        int ispin = i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
                        int jspin = j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_j);
                        int kspin = k + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_k);
                        int lspin = l + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_l);

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


    void Hamiltonian_Heisenberg_Pairs::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
        int nos = spins.size();

        // Set to zero
        hessian.setZero();

        // Single Spin elements
        for (int alpha = 0; alpha < 3; ++alpha)
        {
            for ( int beta = 0; beta < 3; ++beta )
            {
                for (unsigned int i = 0; i < anisotropy_indices.size(); ++i)
                {
                    int idx_i = 3 * anisotropy_indices[i] + alpha;
                    int idx_j = 3 * anisotropy_indices[i] + beta;
                    // scalar x = -2.0*this->anisotropy_magnitudes[i] * std::pow(this->anisotropy_normals[i][alpha], 2);
                    hessian( idx_i, idx_j ) += -2.0 * this->anisotropy_magnitudes[i] *
                                                this->anisotropy_normals[i][alpha] *
                                                this->anisotropy_normals[i][beta];
                }
            }
        }

        // Spin Pair elements
        // Exchange
        for (unsigned int i_pair = 0; i_pair < this->exchange_pairs.size(); ++i_pair)
        {
            for (int da = 0; da < geometry->n_cells[0]; ++da)
            {
                for (int db = 0; db < geometry->n_cells[1]; ++db)
                {
                    for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        if( Vectormath::boundary_conditions_fulfilled( geometry->n_cells, boundary_conditions, translations, exchange_pairs[i_pair].translations ) )
                        {
                            for (int alpha = 0; alpha < 3; ++alpha)
                            {
                                int ispin = 3 * (exchange_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations)) + alpha;
                                int jspin = 3 * (exchange_pairs[i_pair].j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, exchange_pairs[i_pair].translations)) + alpha;
                                hessian(ispin, jspin) += -exchange_magnitudes[i_pair];
                                hessian(jspin, ispin) += -exchange_magnitudes[i_pair];
                            }
                        }
                    }
                }
            }
        }

        // DMI
        for (unsigned int i_pair = 0; i_pair < this->dmi_pairs.size(); ++i_pair)
        {
            for (int da = 0; da < geometry->n_cells[0]; ++da)
            {
                for (int db = 0; db < geometry->n_cells[1]; ++db)
                {
                    for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        if (Vectormath::boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, dmi_pairs[i_pair].translations))
                        {
                            for (int alpha = 0; alpha < 3; ++alpha)
                            {
                                for (int beta = 0; beta < 3; ++beta)
                                {
                                    int ispin = 3 * (dmi_pairs[i_pair].i + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations)) + alpha;
                                    int jspin = 3 * (dmi_pairs[i_pair].j + Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, dmi_pairs[i_pair].translations)) + beta;

                                    if ((alpha == 0 && beta == 1))
                                    {
                                        hessian(ispin, jspin) +=
                                            -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                        hessian(jspin, ispin) +=
                                            -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                    }
                                    else if ((alpha == 1 && beta == 0))
                                    {
                                        hessian(ispin, jspin) +=
                                            dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                        hessian(jspin, ispin) +=
                                            dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                    }
                                    else if ((alpha == 0 && beta == 2))
                                    {
                                        hessian(ispin, jspin) +=
                                            dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                        hessian(jspin, ispin) +=
                                            dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                    }
                                    else if ((alpha == 2 && beta == 0))
                                    {
                                        hessian(ispin, jspin) +=
                                            -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                        hessian(jspin, ispin) +=
                                            -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                    }
                                    else if ((alpha == 1 && beta == 2))
                                    {
                                        hessian(ispin, jspin) +=
                                            -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                        hessian(jspin, ispin) +=
                                            -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                    }
                                    else if ((alpha == 2 && beta == 1))
                                    {
                                        hessian(ispin, jspin) +=
                                            dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                        hessian(jspin, ispin) +=
                                            dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
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
    static const std::string name = "Heisenberg (Pairs)";
    const std::string& Hamiltonian_Heisenberg_Pairs::Name() { return name; }
}

#endif
