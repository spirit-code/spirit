#ifdef SPIRIT_USE_CUDA

#define EIGEN_USE_GPU

#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Core>
#include "FFT.hpp"

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants;
using Engine::Vectormath::cu_check_atom_type;
using Engine::Vectormath::cu_idx_from_pair;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;

namespace Engine
{
    // Construct a Heisenberg Hamiltonian with pairs
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        pairfield exchange_pairs, scalarfield exchange_magnitudes,
        pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
        DDI_Method ddi_method, int ddi_n_periodic_images, scalar ddi_radius,
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
        ddi_method(ddi_method), ddi_n_periodic_images(ddi_n_periodic_images), ddi_cutoff_radius(ddi_radius)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Interactions();
    }

    // Construct a Heisenberg Hamiltonian from shells
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        scalarfield exchange_shell_magnitudes,
        scalarfield dmi_shell_magnitudes, int dmi_shell_chirality,
        DDI_Method ddi_method, int ddi_n_periodic_images, scalar ddi_radius,
        quadrupletfield quadruplets, scalarfield quadruplet_magnitudes,
        std::shared_ptr<Data::Geometry> geometry,
        intfield boundary_conditions
    ) :
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        external_field_magnitude(external_field_magnitude * C::mu_B), external_field_normal(external_field_normal),
        anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
        exchange_pairs_in(0), exchange_magnitudes_in(0), exchange_shell_magnitudes(exchange_shell_magnitudes),
        dmi_pairs_in(0), dmi_magnitudes_in(0), dmi_normals_in(0), dmi_shell_magnitudes(dmi_shell_magnitudes), dmi_shell_chirality(dmi_shell_chirality),
        quadruplets(quadruplets), quadruplet_magnitudes(quadruplet_magnitudes),
        ddi_method(ddi_method), ddi_n_periodic_images(ddi_n_periodic_images), ddi_cutoff_radius(ddi_radius)
    {
        // Generate interaction pairs, constants etc.
        this->Update_Interactions();
    }


    void Hamiltonian_Heisenberg::Update_Interactions()
    {
        // When parallelising (cuda or openmp), we need all neighbours per spin
        const bool use_redundant_neighbours = true;

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
            for (int i = 0; i < dmi_pairs_in.size(); ++i)
            {
                auto& p = dmi_pairs_in[i];
                auto& t = p.translations;
                this->dmi_pairs.push_back(Pair{p.j, p.i, {-t[0], -t[1], -t[2]}});
                this->dmi_magnitudes.push_back(dmi_magnitudes_in[i]);
                this->dmi_normals.push_back(-dmi_normals_in[i]);
            }
        }

        // Dipole-dipole (cutoff)
        scalar radius = this->ddi_cutoff_radius;
        if( this->ddi_method != DDI_Method::Cutoff )
            radius = 0;
        this->ddi_pairs      = Engine::Neighbours::Get_Pairs_in_Radius(*this->geometry, radius);
        this->ddi_magnitudes = scalarfield(this->ddi_pairs.size());
        this->ddi_normals    = vectorfield(this->ddi_pairs.size());

        for (unsigned int i = 0; i < this->ddi_pairs.size(); ++i)
        {
            Engine::Neighbours::DDI_from_Pair(
                *this->geometry,
                { this->ddi_pairs[i].i, this->ddi_pairs[i].j, this->ddi_pairs[i].translations },
                this->ddi_magnitudes[i], this->ddi_normals[i]);
        }
        // Dipole-dipole (FFT)
        this->Prepare_DDI();

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Heisenberg::Update_Energy_Contributions()
    {
        this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

        // External field
        if( this->external_field_magnitude > 0 )
        {
            this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
            this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_zeeman = -1;
        // Anisotropy
        if( this->anisotropy_indices.size() > 0 )
        {
            this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0) });
            this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_anisotropy = -1;
        // Exchange
        if( this->exchange_pairs.size() > 0 )
        {
            this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
            this->idx_exchange = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_exchange = -1;
        // DMI
        if( this->dmi_pairs.size() > 0 )
        {
            this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
            this->idx_dmi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_dmi = -1;
        // Dipole-Dipole
        if( this->ddi_method != DDI_Method::None )
        {
            this->energy_contributions_per_spin.push_back({"DDI", scalarfield(0) });
            this->idx_ddi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_ddi = -1;
        // Quadruplets
        if( this->quadruplets.size() > 0 )
        {
            this->energy_contributions_per_spin.push_back({"Quadruplets", scalarfield(0) });
            this->idx_quadruplet = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_quadruplet = -1;
    }


    void Hamiltonian_Heisenberg::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {
        int nos = spins.size();
        for (auto& pair : contributions)
        {
            // Allocate if not already allocated
            if (pair.second.size() != nos) pair.second = scalarfield(nos, 0);
            // Otherwise set to zero
            else for (auto& pair : contributions) Vectormath::fill(pair.second, 0);
        }

        // External field
        if( this->idx_zeeman >=0 )     E_Zeeman(spins, contributions[idx_zeeman].second);

        // Anisotropy
        if( this->idx_anisotropy >=0 ) E_Anisotropy(spins, contributions[idx_anisotropy].second);

        // Exchange
        if( this->idx_exchange >=0 )   E_Exchange(spins, contributions[idx_exchange].second);
        // DMI
        if( this->idx_dmi >=0 )        E_DMI(spins,contributions[idx_dmi].second);
        // DDI
        if( this->idx_ddi >=0 )        E_DDI(spins, contributions[idx_ddi].second);

        // Quadruplets
        if (this->idx_quadruplet >=0 ) E_Quadruplet(spins, contributions[idx_quadruplet].second);
    }


    __global__ void CU_E_Zeeman(const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude, const Vector3 external_field_normal, scalar * Energy, size_t n_cells_total)
    {
        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < n_cells_total;
            icell +=  blockDim.x * gridDim.x)
        {
            for (int ibasis=0; ibasis<n_cell_atoms; ++ibasis)
            {
                int ispin = icell + ibasis;
                if ( cu_check_atom_type(atom_types[ispin]) )
                    Energy[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal.dot(spins[ispin]);
            }
        }
    }
    void Hamiltonian_Heisenberg::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        int size = geometry->n_cells_total;
        CU_E_Zeeman<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(), this->external_field_magnitude, this->external_field_normal, Energy.data(), size);
        CU_CHECK_AND_SYNC();
    }


    __global__ void CU_E_Anisotropy(const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies, const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, scalar * Energy, size_t n_cells_total)
    {
        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < n_cells_total;
            icell +=  blockDim.x * gridDim.x)
        {
            for (int iani=0; iani<n_anisotropies; ++iani)
            {
                int ispin = icell*n_cell_atoms + anisotropy_indices[iani];
                if ( cu_check_atom_type(atom_types[ispin]) )
                    Energy[ispin] -= anisotropy_magnitude[iani] * std::pow(anisotropy_normal[iani].dot(spins[ispin]), 2.0);
            }
        }
    }
    void Hamiltonian_Heisenberg::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
    {
        int size = geometry->n_cells_total;
        CU_E_Anisotropy<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, this->anisotropy_indices.size(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), Energy.data(), size);
        CU_CHECK_AND_SYNC();
    }


    __global__ void CU_E_Exchange(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_cell_atoms,
            int n_pairs, const Pair * pairs, const scalar * magnitudes, scalar * Energy, size_t size)
    {
        int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
        int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < size;
            icell +=  blockDim.x * gridDim.x)
        {
            for(auto ipair = 0; ipair < n_pairs; ++ipair)
            {
                int ispin = pairs[ipair].i + icell*n_cell_atoms;
                int jspin = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, pairs[ipair]);
                if (jspin >= 0)
                {
                    Energy[ispin] -= 0.5 * magnitudes[ipair] * spins[ispin].dot(spins[jspin]);
                }
            }
        }
    }
    void Hamiltonian_Heisenberg::E_Exchange(const vectorfield & spins, scalarfield & Energy)
    {
        int size = geometry->n_cells_total;
        CU_E_Exchange<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
                this->exchange_pairs.size(), this->exchange_pairs.data(), this->exchange_magnitudes.data(), Energy.data(), size);
        CU_CHECK_AND_SYNC();
    }


    __global__ void CU_E_DMI(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_cell_atoms,
            int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals, scalar * Energy, size_t size)
    {
        int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
        int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < size;
            icell +=  blockDim.x * gridDim.x)
        {
            for(auto ipair = 0; ipair < n_pairs; ++ipair)
            {
                int ispin = pairs[ipair].i + icell*n_cell_atoms;
                int jspin = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, pairs[ipair]);
                if (jspin >= 0)
                {
                    Energy[ispin] -= 0.5 * magnitudes[ipair] * normals[ipair].dot(spins[ispin].cross(spins[jspin]));
                }
            }
        }
    }
    void Hamiltonian_Heisenberg::E_DMI(const vectorfield & spins, scalarfield & Energy)
    {
        int size = geometry->n_cells_total;
        CU_E_DMI<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
                this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), Energy.data(), size);
        CU_CHECK_AND_SYNC();
    }


    void Hamiltonian_Heisenberg::E_DDI(const vectorfield & spins, scalarfield & Energy)
    {
        if( this->ddi_method == DDI_Method::FFT )
            this->E_DDI_FFT(spins, Energy);
        else if( this->ddi_method == DDI_Method::Cutoff )
            this->E_DDI_Cutoff(spins, Energy);
    }

    void Hamiltonian_Heisenberg::E_DDI_Cutoff(const vectorfield & spins, scalarfield & Energy)
    {
        // //scalar mult = -mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
        // scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        // // scalar result = 0.0;

        // for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
        // {
        //     if (ddi_magnitudes[i_pair] > 0.0)
        //     {
        //         for (int da = 0; da < geometry->n_cells[0]; ++da)
        //         {
        //             for (int db = 0; db < geometry->n_cells[1]; ++db)
        //             {
        //                 for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
        //                 {
        //                     std::array<int, 3 > translations = { da, db, dc };
        //                     // int idx_i = ddi_pairs[i_pair].i;
        //                     // int idx_j = ddi_pairs[i_pair].j;
        //                     int idx_i = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
        //                     int idx_j = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, ddi_pairs[i_pair].translations);
        //                     Energy[idx_i] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
        //                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) - spins[idx_i].dot(spins[idx_j]));
        //                     Energy[idx_j] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
        //                         (3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) - spins[idx_i].dot(spins[idx_j]));
        //                 }
        //             }
        //         }
        //     }
        // }
    }// end DipoleDipole

    // TODO: add dot_scaled to Vectormath and use that
    __global__ void CU_E_DDI_FFT(scalar * Energy, const Vector3 * spins, const Vector3 * gradients , const int nos, const int n_cell_atoms, const scalar * mu_s)
    {
        for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < nos; idx += blockDim.x * gridDim.x)
        {
            Energy[idx] += 0.5 * mu_s[idx] * spins[idx].dot(gradients[idx]);
        }
    }
    void Hamiltonian_Heisenberg::E_DDI_FFT(const vectorfield & spins, scalarfield & Energy)
    {
        //todo maybe the gradient should be cached somehow, it is quite inefficient to calculate it
        //again just for the energy
        vectorfield gradients_temp(geometry->nos);
        Vectormath::fill(gradients_temp, {0,0,0});
        this->Gradient_DDI(spins, gradients_temp);
        CU_E_DDI_FFT<<<(geometry->nos + 1023)/1024, 1024>>>(Energy.data(), spins.data(), gradients_temp.data(), geometry->nos, geometry->n_cell_atoms, geometry->mu_s.data());
    }// end DipoleDipole


    void Hamiltonian_Heisenberg::E_Quadruplet(const vectorfield & spins, scalarfield & Energy)
    {
        // for (unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad)
        // {
        //     for (int da = 0; da < geometry->n_cells[0]; ++da)
        //     {
        //         for (int db = 0; db < geometry->n_cells[1]; ++db)
        //         {
        //             for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
        //             {
        //                 std::array<int, 3 > translations = { da, db, dc };
        //                 // int i = quadruplets[iquad].i;
        //                 // int j = quadruplets[iquad].j;
        //                 // int k = quadruplets[iquad].k;
        //                 // int l = quadruplets[iquad].l;
        //                 int i = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
        //                 int j = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j);
        //                 int k = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k);
        //                 int l = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l);
        //                 Energy[i] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        //                 Energy[j] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        //                 Energy[k] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        //                 Energy[l] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        //             }
        //         }
        //     }
        // }
    }


    scalar Hamiltonian_Heisenberg::Energy_Single_Spin(int ispin_in, const vectorfield & spins)
    {
        int icell  = ispin_in / this->geometry->n_cell_atoms;
        int ibasis = ispin_in - icell*this->geometry->n_cell_atoms;
        scalar Energy = 0;

        // External field
        if (this->idx_zeeman >= 0)
        {
            if (check_atom_type(this->geometry->atom_types[ispin_in]))
                Energy -= geometry->mu_s[ispin_in] * this->external_field_magnitude * this->external_field_normal.dot(spins[ispin_in]);
        }

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
                    const scalar mult = 0.5 * geometry->mu_s[ispin] * geometry->mu_s[jspin]
                        * C::mu_0 * std::pow(C::mu_B, 2) / ( 4*C::Pi * 1e-30 );

                    if (jspin >= 0)
                    {
                        Energy -= mult / std::pow(this->ddi_magnitudes[ipair], 3.0) *
                            (3 * spins[ispin].dot(this->ddi_normals[ipair]) * spins[ispin].dot(this->ddi_normals[ipair]) - spins[ispin].dot(spins[ispin]));
                    }
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
            }
        }

        return Energy;
    }


    void Hamiltonian_Heisenberg::Gradient(const vectorfield & spins, vectorfield & gradient)
    {
        // Set to zero
        Vectormath::fill(gradient, {0,0,0});

        // External field
        this->Gradient_Zeeman(gradient);

        // Anisotropy
        this->Gradient_Anisotropy(spins, gradient);

        // Pairs
        //    Exchange
        this->Gradient_Exchange(spins, gradient);
        //    DMI
        this->Gradient_DMI(spins, gradient);
        //    DDI
        this->Gradient_DDI(spins, gradient);
        //    Quadruplet
        this->Gradient_Quadruplet(spins, gradient);
    }


    __global__ void CU_Gradient_Zeeman( const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude, const Vector3 external_field_normal, Vector3 * gradient, size_t n_cells_total)
    {
        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < n_cells_total;
            icell +=  blockDim.x * gridDim.x)
        {
            for (int ibasis=0; ibasis<n_cell_atoms; ++ibasis)
            {
                int ispin = icell + ibasis;
                if ( cu_check_atom_type(atom_types[ispin]) )
                    gradient[ispin] -= mu_s[ispin] * external_field_magnitude*external_field_normal;
            }
        }
    }
    void Hamiltonian_Heisenberg::Gradient_Zeeman(vectorfield & gradient)
    {
        int size = geometry->n_cells_total;
        CU_Gradient_Zeeman<<<(size+1023)/1024, 1024>>>( this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(), this->external_field_magnitude, this->external_field_normal, gradient.data(), size );
        CU_CHECK_AND_SYNC();
    }


    __global__ void CU_Gradient_Anisotropy(const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const int n_anisotropies, const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, Vector3 * gradient, size_t n_cells_total)
    {
        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < n_cells_total;
            icell +=  blockDim.x * gridDim.x)
        {
            for (int iani=0; iani<n_anisotropies; ++iani)
            {
                int ispin = icell*n_cell_atoms + anisotropy_indices[iani];
                if ( cu_check_atom_type(atom_types[ispin]) )
                {
                    scalar sc = -2 * anisotropy_magnitude[iani] * anisotropy_normal[iani].dot(spins[ispin]);
                    gradient[ispin] += sc*anisotropy_normal[iani];
                }
            }
        }
    }
    void Hamiltonian_Heisenberg::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
        int size = geometry->n_cells_total;
        CU_Gradient_Anisotropy<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, this->anisotropy_indices.size(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), gradient.data(), size );
        CU_CHECK_AND_SYNC();
    }


    __global__ void CU_Gradient_Exchange(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_cell_atoms,
            int n_pairs, const Pair * pairs, const scalar * magnitudes, Vector3 * gradient, size_t size)
    {
        int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
        int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < size;
            icell +=  blockDim.x * gridDim.x)
        {
            for(auto ipair = 0; ipair < n_pairs; ++ipair)
            {
                int ispin = pairs[ipair].i + icell*n_cell_atoms;
                int jspin = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, pairs[ipair]);
                if (jspin >= 0)
                {
                    gradient[ispin] -= magnitudes[ipair]*spins[jspin];
                }
            }
        }
    }
    void Hamiltonian_Heisenberg::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        int size = geometry->n_cells_total;
        CU_Gradient_Exchange<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
                this->exchange_pairs.size(), this->exchange_pairs.data(), this->exchange_magnitudes.data(), gradient.data(), size );
        CU_CHECK_AND_SYNC();
    }


    __global__ void CU_Gradient_DMI(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_cell_atoms,
            int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals, Vector3 * gradient, size_t size)
    {
        int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
        int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

        for(auto icell = blockIdx.x * blockDim.x + threadIdx.x;
            icell < size;
            icell +=  blockDim.x * gridDim.x)
        {
            for(auto ipair = 0; ipair < n_pairs; ++ipair)
            {
                int ispin = pairs[ipair].i + icell*n_cell_atoms;
                int jspin = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, pairs[ipair]);
                if (jspin >= 0)
                {
                    gradient[ispin] -= magnitudes[ipair]*spins[jspin].cross(normals[ipair]);
                }
            }
        }
    }
    void Hamiltonian_Heisenberg::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        int size = geometry->n_cells_total;
        CU_Gradient_DMI<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
                this->dmi_pairs.size(),  this->dmi_pairs.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), gradient.data(), size );
        CU_CHECK_AND_SYNC();
    }

    void Hamiltonian_Heisenberg::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
    {
        if( this->ddi_method == DDI_Method::FFT )
            this->Gradient_DDI_FFT(spins, gradient);
        else if( this->ddi_method == DDI_Method::Cutoff )
            this->Gradient_DDI_Cutoff(spins, gradient);
    }

    void Hamiltonian_Heisenberg::Gradient_DDI_Cutoff(const vectorfield & spins, vectorfield & gradient)
    {
        // TODO
    }

    __global__ void CU_FFT_Pointwise_Mult(Matrix3c * fmatrices, FFT::FFT_cpx_type * fspins, FFT::FFT_cpx_type * res, int * n_cells_padded)
    {
        int N = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];
        auto mapSpins = Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                        (reinterpret_cast<std::complex<scalar>*>(NULL),
                        Eigen::Stride<1,Eigen::Dynamic>(1,N));
        auto mapResult = Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                        (reinterpret_cast<std::complex<scalar>*>(NULL),
                        Eigen::Stride<1,Eigen::Dynamic>(1,N));

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;


        int t1 = n_cells_padded[0]/2 + 1;
        int t2 = n_cells_padded[0] - t1;
        // for(int i=index; i < n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2]; i+=stride)
        for(int i=index; i < (n_cells_padded[0]/2+1) * n_cells_padded[1] * n_cells_padded[2]; i+=stride)
        {
            int idx = i + i/t1 * t2;
            auto& D_mat = fmatrices[idx];
            new (&mapSpins) Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                (reinterpret_cast<std::complex<scalar>*>(fspins + idx),
                Eigen::Stride<1,Eigen::Dynamic>(1,N));

            new (&mapResult) Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                    (reinterpret_cast<std::complex<scalar>*>(res + idx),
                    Eigen::Stride<1,Eigen::Dynamic>(1,N));

            mapResult += D_mat * mapSpins;
        }
    }

    __global__ void CU_Write_FFT_Gradients(FFT::FFT_real_type * res_mult, Vector3 * gradient, int * n_cells_padded, int * n_cells, int nbasis)
    {
        int n_write = n_cells[0] * n_cells[1] * n_cells[2];
        int sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for(int i=index; i<n_write; i+=stride)
        {
            int idx1 =  i * nbasis;

            int c = i / (n_cells[0] * n_cells[1]);
            int b = (i - c * n_cells[1] * n_cells[0]) / n_cells[0];
            int a = (i - c * n_cells[1] * n_cells[0] - b * n_cells[0]);

            int idx2 = a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1];
            // int idx2 = (a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1]) * 3 * nbasis + 3 * ibasis;

            for(int j = 0; j<3; j++)
            {
                gradient[idx1][j] -= (res_mult[idx2 + j * sublattice_size] / sublattice_size);
            }
        }
    }

    void Hamiltonian_Heisenberg::Gradient_DDI_FFT(const vectorfield & spins, vectorfield & gradient)
    {
        // Size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        FFT_spins(spins);

        // auto& ft_D_matrices = fft_plan_d.cpx_ptr;
        auto& ft_spins = fft_plan_spins.cpx_ptr;

        auto& res_iFFT = fft_plan_rev.real_ptr;
        auto& res_mult = fft_plan_rev.cpx_ptr;

        auto mapSpins = Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                        (reinterpret_cast<std::complex<scalar>*>(NULL),
                        Eigen::Stride<1,Eigen::Dynamic>(1,sublattice_size));

        auto mapResult = Eigen::Map<Vector3c, 0, Eigen::Stride<1,Eigen::Dynamic> >
                        (reinterpret_cast<std::complex<scalar>*>(NULL),
                        Eigen::Stride<1,Eigen::Dynamic>(1,sublattice_size));

        // Loop over basis atoms (i.e sublattices)
        for(int i_b1 = 0; i_b1 < geometry->n_cell_atoms; ++i_b1)
        {
            std::fill(fft_plan_rev.cpx_ptr.data(), fft_plan_rev.cpx_ptr.data() + 3 * sublattice_size * geometry->n_cell_atoms, FFT::FFT_cpx_type());
            std::fill(res_iFFT.data(), res_iFFT.data() + 3 * sublattice_size * geometry->n_cell_atoms, 0.0);

            for(int i_b2 = 0; i_b2 < B; ++i_b2)
            {
                // Look up at which position the correct D-matrices are saved
                int b_diff = b_diff_lookup[i_b1 + i_b2 * geometry->n_cell_atoms];
                CU_FFT_Pointwise_Mult<<<((n_cells_padded[0]/2+1) * n_cells_padded[1] * n_cells_padded[2] + 1023 )/1024, 1024>>>(d_mats_ft.data() + b_diff * sublattice_size , ft_spins.data() + i_b2 * 3 * sublattice_size, res_mult.data() + i_b1 * 3 * sublattice_size, n_cells_padded.data());
            }//end iteration over second sublattice
            FFT::batch_iFour_3D(fft_plan_rev);
            CU_Write_FFT_Gradients<<<(geometry->n_cells_total+1023)/1024, 1024 >>>(res_iFFT.data() + i_b1 * 3 * sublattice_size, gradient.data() + i_b1, n_cells_padded.data(), geometry->n_cells.data(), geometry->n_cell_atoms);
        }//end iteration sublattice 1CU_CHECK_AND_SYNC();
    }//end Field_DipoleDipole


    void Hamiltonian_Heisenberg::Gradient_Quadruplet(const vectorfield & spins, vectorfield & gradient)
    {
        // for (unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad)
        // {
        // 	int i = quadruplets[iquad].i;
        // 	int j = quadruplets[iquad].j;
        // 	int k = quadruplets[iquad].k;
        // 	int l = quadruplets[iquad].l;
        // 	for (int da = 0; da < geometry->n_cells[0]; ++da)
        // 	{
        // 		for (int db = 0; db < geometry->n_cells[1]; ++db)
        // 		{
        // 			for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
        // 			{
        // 				std::array<int, 3 > translations = { da, db, dc };
        // 				int ispin = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
        // 				int jspin = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j);
        // 				int kspin = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k);
        // 				int lspin = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l);
        // 				gradient[ispin] -= quadruplet_magnitudes[iquad] * spins[jspin] * (spins[kspin].dot(spins[lspin]));
        // 				gradient[jspin] -= quadruplet_magnitudes[iquad] * spins[ispin] * (spins[kspin].dot(spins[lspin]));
        // 				gradient[kspin] -= quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * spins[lspin];
        // 				gradient[lspin] -= quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * spins[kspin];
        // 			}
        // 		}
        // 	}
        // }
    }


    void Hamiltonian_Heisenberg::Hessian(const vectorfield & spins, MatrixX & hessian)
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
        for (unsigned int i_pair = 0; i_pair < this->exchange_pairs.size(); ++i_pair)
        {
            for (int da = 0; da < geometry->n_cells[0]; ++da)
            {
                for (int db = 0; db < geometry->n_cells[1]; ++db)
                {
                    for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
                    {
                        std::array<int, 3 > translations = { da, db, dc };
                        for (int alpha = 0; alpha < 3; ++alpha)
                        {
                            // int idx_i = 3 * exchange_pairs[i_pair].i + alpha;
                            // int idx_j = 3 * exchange_pairs[i_pair].j + alpha;
                            int idx_i = 3 * Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations) + alpha;
                            int idx_j = 3 * Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, exchange_pairs[i_pair].translations) + alpha;
                            hessian(idx_i, idx_j) += -exchange_magnitudes[i_pair];
                            hessian(idx_j, idx_i) += -exchange_magnitudes[i_pair];
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
                        for (int alpha = 0; alpha < 3; ++alpha)
                        {
                            for (int beta = 0; beta < 3; ++beta)
                            {
                                // int idx_i = 3 * dmi_pairs[i_pair].i + alpha;
                                // int idx_j = 3 * dmi_pairs[i_pair].j + beta;
                                int idx_i = 3 * Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations) + alpha;
                                int idx_j = 3 * Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, dmi_pairs[i_pair].translations) + alpha;
                                if ((alpha == 0 && beta == 1))
                                {
                                    hessian(idx_i, idx_j) +=
                                        -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                    hessian(idx_j, idx_i) +=
                                        -dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                }
                                else if ((alpha == 1 && beta == 0))
                                {
                                    hessian(idx_i, idx_j) +=
                                        dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                    hessian(idx_j, idx_i) +=
                                        dmi_magnitudes[i_pair] * dmi_normals[i_pair][2];
                                }
                                else if ((alpha == 0 && beta == 2))
                                {
                                    hessian(idx_i, idx_j) +=
                                        dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                    hessian(idx_j, idx_i) +=
                                        dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                }
                                else if ((alpha == 2 && beta == 0))
                                {
                                    hessian(idx_i, idx_j) +=
                                        -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                    hessian(idx_j, idx_i) +=
                                        -dmi_magnitudes[i_pair] * dmi_normals[i_pair][1];
                                }
                                else if ((alpha == 1 && beta == 2))
                                {
                                    hessian(idx_i, idx_j) +=
                                        -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                    hessian(idx_j, idx_i) +=
                                        -dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                }
                                else if ((alpha == 2 && beta == 1))
                                {
                                    hessian(idx_i, idx_j) +=
                                        dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                    hessian(idx_j, idx_i) +=
                                        dmi_magnitudes[i_pair] * dmi_normals[i_pair][0];
                                }
                            }
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
        //		* geometry->cell_mu_s[idx_1] * geometry->cell_mu_s[idx_2]
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
        // Size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        // Bravais vectors
        Vector3 ta = geometry->bravais_vectors[0];
        Vector3 tb = geometry->bravais_vectors[1];
        Vector3 tc = geometry->bravais_vectors[2];

        auto& fft_spin_inputs = fft_plan_spins.real_ptr;

        //iterate over the **original** system
        for(int c = 0; c < Nc; ++c)
        {
            for(int b = 0; b < Nb; ++b)
            {
                for(int a = 0; a < Na; ++a)
                {
                    for(int bi = 0; bi < B; ++bi)
                    {
                        int idx_pad = a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1] + 3 * sublattice_size * bi;
                        // int idx_pad = (a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1]) * 3 * geometry->n_cell_atoms + 3 * bi;
                        int idx_orig = bi + a * geometry->n_cell_atoms + b * Na * geometry->n_cell_atoms  + c * Na * Nb * B;

                        fft_spin_inputs[idx_pad        ] = spins[idx_orig][0] * geometry->mu_s[bi];
                        fft_spin_inputs[idx_pad + 1 * sublattice_size] = spins[idx_orig][1] * geometry->mu_s[bi];
                        fft_spin_inputs[idx_pad + 2 * sublattice_size] = spins[idx_orig][2] * geometry->mu_s[bi];
                    }
                }
            }
        }
        FFT::batch_Four_3D(fft_plan_spins);
    }

    void Hamiltonian_Heisenberg::FFT_Dipole_Mats(std::array<int, 3> pb_images)
    {
        // Prefactor of ddi interaction
        scalar mult = 2 * C::mu_0 * C::mu_B * C::mu_B  / ( 4*C::Pi * 1e-30 );

        // Size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        // Bravais vectors
        Vector3 ta = geometry->bravais_vectors[0];
        Vector3 tb = geometry->bravais_vectors[1];
        Vector3 tc = geometry->bravais_vectors[2];

        auto& fft_dipole_inputs = fft_plan_d.real_ptr;

        int count = -1;
        // Loop over basis atoms (i.e sublattices)
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
                for(int c = 0; c < n_cells_padded[2]; ++c)
                {
                    for(int b = 0; b < n_cells_padded[1]; ++b)
                    {
                        for(int a = 0; a < n_cells_padded[0]; ++a)
                        {
                            int a_idx = a < Na ? a : a - n_cells_padded[0];
                            int b_idx = b < Nb ? b : b - n_cells_padded[1];
                            int c_idx = c < Nc ? c : c - n_cells_padded[2];

                            Vector3 diff =    (a_idx + geometry->cell_atoms[i_b1][0] - geometry->cell_atoms[i_b2][0]) * ta
                                            + (b_idx + geometry->cell_atoms[i_b1][1] - geometry->cell_atoms[i_b2][1]) * tb
                                            + (c_idx + geometry->cell_atoms[i_b1][2] - geometry->cell_atoms[i_b2][2]) * tc;

                            if(!(a==0 && b==0 && c==0 && i_b1 == i_b2))
                            {
                                auto d = diff.norm();
                                auto d3 = d * d * d;
                                auto d5 = d * d * d * d * d;
                                scalar Dxx = mult * (3 * diff[0]*diff[0] / d5 - 1/d3);
                                scalar Dxy = mult *  3 * diff[0]*diff[1] / d5;          //same as Dyx
                                scalar Dxz = mult *  3 * diff[0]*diff[2] / d5;          //same as Dzx
                                scalar Dyy = mult * (3 * diff[1]*diff[1] / d5 - 1/d3);
                                scalar Dyz = mult *  3 * diff[1]*diff[2] / d5;          //same as Dzy
                                scalar Dzz = mult * (3 * diff[2]*diff[2] / d5 - 1/d3);

                                int idx_pad = a + b * n_cells_padded[0] + c * n_cells_padded[1] * n_cells_padded[0] + 6 * sublattice_size * count;

                                // int idx_pad = (a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1]) * 6 * symmetry_count + 6 * (count-1);

                                fft_dipole_inputs[idx_pad        ] = Dxx;
                                fft_dipole_inputs[idx_pad + 1 * sublattice_size] = Dxy;
                                fft_dipole_inputs[idx_pad + 2 * sublattice_size] = Dxz;
                                fft_dipole_inputs[idx_pad + 3 * sublattice_size] = Dyy;
                                fft_dipole_inputs[idx_pad + 4 * sublattice_size] = Dyz;
                                fft_dipole_inputs[idx_pad + 5 * sublattice_size] = Dzz;
                            }
                        }
                    }
                }
            }
        }
        FFT::batch_Four_3D(fft_plan_d);
    }

    void Hamiltonian_Heisenberg::Prepare_DDI(std::array<int, 3> pb_images)
    {
        n_cells_padded.resize(3);
        n_cells_padded[0] = (geometry->n_cells[0] > 1) ? 2 * geometry->n_cells[0] : 1;
        n_cells_padded[1] = (geometry->n_cells[1] > 1) ? 2 * geometry->n_cells[1] : 1;
        n_cells_padded[2] = (geometry->n_cells[2] > 1) ? 2 * geometry->n_cells[2] : 1;
        sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

        b_diff_lookup.resize(geometry->n_cell_atoms * geometry->n_cell_atoms);

        //Ee dont need to transform over length 1 dims
        std::vector<int> fft_dims;
        for(int i = 2; i >= 0; i--) //notice reverse order is important!
        {
            if(n_cells_padded[i] > 1)
                fft_dims.push_back(n_cells_padded[i]);
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
        fft_plan_d.real_ptr = field<FFT::FFT_real_type>(symmetry_count * 6 * sublattice_size);
        fft_plan_d.cpx_ptr  = field<FFT::FFT_cpx_type>(symmetry_count * 6 * sublattice_size);
        fft_plan_d.CreateConfiguration();

        fft_plan_spins.dims     = fft_dims;
        fft_plan_spins.inverse  = false;
        fft_plan_spins.howmany  = 3 * geometry->n_cell_atoms;
        fft_plan_spins.real_ptr = field<FFT::FFT_real_type>(3 * sublattice_size * geometry->n_cell_atoms);
        fft_plan_spins.cpx_ptr  = field<FFT::FFT_cpx_type>(3 * sublattice_size * geometry->n_cell_atoms);
        fft_plan_spins.CreateConfiguration();

        fft_plan_rev.dims     = fft_dims;
        fft_plan_rev.inverse  = true;
        fft_plan_rev.howmany  = 3 * geometry->n_cell_atoms;
        fft_plan_rev.cpx_ptr  = field<FFT::FFT_cpx_type>(3 * sublattice_size * geometry->n_cell_atoms);
        fft_plan_rev.real_ptr = field<FFT::FFT_real_type>(3 * sublattice_size * geometry->n_cell_atoms);
        fft_plan_rev.CreateConfiguration();

        //perform FFT of dipole matrices
        FFT_Dipole_Mats(pb_images);
        d_mats_ft = field<Matrix3c>(sublattice_size * symmetry_count);

        //Write out ft dipole matrices
        for(int c = 0; c < n_cells_padded[2]; ++c)
        {
            for(int b = 0; b < n_cells_padded[1]; ++b)
            {
                for(int a = 0; a < n_cells_padded[0]; ++a)
                {
                    for(int b_diff = 0; b_diff < symmetry_count; ++b_diff)
                    {
                        // int idx = (a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1]) * 6 * symmetry_count + 6 * b_diff;
                        int idx = (a + b * n_cells_padded[0] + c * n_cells_padded[0] * n_cells_padded[1]) + 6 * sublattice_size * b_diff;

                        auto fD_xx = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx        ]);
                        auto fD_xy = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + 1 * sublattice_size]);
                        auto fD_xz = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + 2 * sublattice_size]);
                        auto fD_yy = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + 3 * sublattice_size]);
                        auto fD_yz = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + 4 * sublattice_size]);
                        auto fD_zz = reinterpret_cast<std::complex<scalar>* >(&fft_plan_d.cpx_ptr[idx + 5 * sublattice_size]);
                        d_mats_ft[idx] <<   *fD_xx, *fD_xy, *fD_xz,
                                            *fD_xy, *fD_yy, *fD_yz,
                                            *fD_xz, *fD_yz, *fD_zz;
                    }
                }
            }
        }
    }//end prepare

    // Hamiltonian name as string
    static const std::string name = "Heisenberg";
    const std::string& Hamiltonian_Heisenberg::Name() { return name; }
}

#endif