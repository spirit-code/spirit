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
using Engine::Vectormath::tupel_from_idx;

namespace Engine
{
    // Construct a Heisenberg Hamiltonian with pairs
    Hamiltonian_Heisenberg::Hamiltonian_Heisenberg(
        scalar external_field_magnitude, Vector3 external_field_normal,
        intfield anisotropy_indices, scalarfield anisotropy_magnitudes, vectorfield anisotropy_normals,
        pairfield exchange_pairs, scalarfield exchange_magnitudes,
        pairfield dmi_pairs, scalarfield dmi_magnitudes, vectorfield dmi_normals,
        DDI_Method ddi_method, intfield ddi_n_periodic_images, scalar ddi_radius,
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
        DDI_Method ddi_method, intfield ddi_n_periodic_images, scalar ddi_radius,
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
  
        // === DEBUG: begin gradient comparison ===
            // vectorfield gradients_temp_dir;
            // gradients_temp_dir.resize(this->geometry->nos);
            // Vectormath::fill(gradients_temp_dir, {0,0,0});
            // Gradient_DDI_Direct(spins, gradients_temp_dir);

            // //get deviation
            // std::array<scalar, 3> deviation = {0,0,0};
            // std::array<scalar, 3> avg = {0,0,0};
            // std::array<scalar, 3> avg_ft = {0,0,0};

            // for(int i = 0; i < this->geometry->nos; i++)
            // {
            //     for(int d = 0; d < 3; d++)
            //     {
            //         deviation[d] += std::pow(gradients_temp[i][d] - gradients_temp_dir[i][d], 2);
            //         avg[d] += gradients_temp_dir[i][d];
            //         avg_ft[d] += gradients_temp[i][d];
            //     }
            // }
            // std::cerr << "Avg. Gradient (Direct) = " << avg[0]/this->geometry->nos << " " << avg[1]/this->geometry->nos << " " << avg[2]/this->geometry->nos << std::endl;
            // std::cerr << "Avg. Gradient (FFT)    = " << avg_ft[0]/this->geometry->nos << " " << avg_ft[1]/this->geometry->nos << " " << avg_ft[2]/this->geometry->nos << std::endl;
            // std::cerr << "Ratio                  = " << avg_ft[0]/avg[0] << " " << avg_ft[1]/avg[1] << " " << avg_ft[2]/avg[2] << std::endl;            
            // std::cerr << "Avg. Deviation         = " << deviation[0]/this->geometry->nos << " " << deviation[1]/this->geometry->nos << " " << deviation[2]/this->geometry->nos << std::endl;
            // std::cerr << " ---------------- " << std::endl;
        // ==== DEBUG: end gradient comparison ====

    }// end DipoleDipole
    void Hamiltonian_Heisenberg::Gradient_DDI_Direct(const vectorfield & spins, vectorfield & gradient)
    {
        scalar mult = 2 * C::mu_0 * C::mu_B * C::mu_B / ( 4*C::Pi * 1e-30 );
        scalar d, d3, d5, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz;
        Vector3 diff;

        for(int idx1 = 0; idx1 < geometry->nos; idx1++)
        {
            for(int idx2 = 0; idx2 < geometry->nos; idx2++)
            {
                if(idx1 != idx2)
                {
                    auto& m2 = spins[idx2];

                    diff = this->geometry->positions[idx2] - this->geometry->positions[idx1];
                    d = diff.norm();
                    d3 = d * d * d;
                    d5 = d * d * d * d * d;
                    Dxx = mult * (3 * diff[0]*diff[0] / d5 - 1/d3);
                    Dxy = mult *  3 * diff[0]*diff[1] / d5;          //same as Dyx
                    Dxz = mult *  3 * diff[0]*diff[2] / d5;          //same as Dzx
                    Dyy = mult * (3 * diff[1]*diff[1] / d5 - 1/d3);
                    Dyz = mult *  3 * diff[1]*diff[2] / d5;          //same as Dzy
                    Dzz = mult * (3 * diff[2]*diff[2] / d5 - 1/d3);

                    gradient[idx1][0] -= ((Dxx * m2[0] + Dxy * m2[1] + Dxz * m2[2]) * geometry->mu_s[idx2]);
                    gradient[idx1][1] -= ((Dxy * m2[0] + Dyy * m2[1] + Dyz * m2[2]) * geometry->mu_s[idx2]);
                    gradient[idx1][2] -= ((Dxz * m2[0] + Dyz * m2[1] + Dzz * m2[2]) * geometry->mu_s[idx2]);
                }
            }
        }
    }

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


    __global__ void CU_FFT_Pointwise_Mult(FFT::FFT_cpx_type * ft_D_matrices, FFT::FFT_cpx_type * ft_spins, FFT::FFT_cpx_type * res_mult, int* iteration_bounds, int i_b1, int* b_diff_lookup, FFT::StrideContainer d_stride, FFT::StrideContainer spin_stride)
    {
        int n = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
        int tupel[4];
        int idx_b1, idx_b2, idx_d, b_diff;

        for(int ispin = blockIdx.x * blockDim.x + threadIdx.x; ispin < n; ispin += blockDim.x * gridDim.x)
        {
            tupel_from_idx(ispin, tupel, iteration_bounds, 4); // tupel now is {i_b2, a, b, c}

            b_diff = b_diff_lookup[i_b1 + tupel[0] * iteration_bounds[0]];
            idx_b1 = i_b1 * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
            idx_b2 = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
            idx_d  = b_diff * d_stride.basis + tupel[1] * d_stride.a + tupel[2] * d_stride.b + tupel[3] * d_stride.c;

            auto& fs_x = ft_spins[idx_b2                       ];
            auto& fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
            auto& fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

            auto& fD_xx = ft_D_matrices[idx_d                    ];
            auto& fD_xy = ft_D_matrices[idx_d + 1 * d_stride.comp];
            auto& fD_xz = ft_D_matrices[idx_d + 2 * d_stride.comp];
            auto& fD_yy = ft_D_matrices[idx_d + 3 * d_stride.comp];
            auto& fD_yz = ft_D_matrices[idx_d + 4 * d_stride.comp];
            auto& fD_zz = ft_D_matrices[idx_d + 5 * d_stride.comp];

            FFT::addTo(res_mult[idx_b1                       ], FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z), tupel[0] == 0);
            FFT::addTo(res_mult[idx_b1 + 1 * spin_stride.comp], FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z), tupel[0] == 0);
            FFT::addTo(res_mult[idx_b1 + 2 * spin_stride.comp], FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z), tupel[0] == 0);
        }
    }

    __global__ void CU_Write_FFT_Gradients(FFT::FFT_real_type * resiFFT, Vector3 * gradient, FFT::StrideContainer spin_stride , int * iteration_bounds, int n_cell_atoms, int bi, scalar * mu_s, int sublattice_size)
    {
        int n_write = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2];
        int tupel[3];
        int idx_pad;
        int idx_orig;
        for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_write; idx += blockDim.x * gridDim.x)
        {
            tupel_from_idx(idx, tupel, iteration_bounds, 3); //tupel now is {a, b, c}
            idx_orig = bi + n_cell_atoms * idx;
            idx_pad = bi * spin_stride.basis + tupel[0] * spin_stride.a + tupel[1] * spin_stride.b + tupel[2] * spin_stride.c;
            gradient[idx_orig][0] -= resiFFT[idx_pad                       ] * mu_s[idx_orig] / sublattice_size;
            gradient[idx_orig][1] -= resiFFT[idx_pad + 1 * spin_stride.comp] * mu_s[idx_orig] / sublattice_size;
            gradient[idx_orig][2] -= resiFFT[idx_pad + 2 * spin_stride.comp] * mu_s[idx_orig] / sublattice_size;
            }
        }

    void Hamiltonian_Heisenberg::Gradient_DDI_FFT(const vectorfield & spins, vectorfield & gradient)
    {
        // Size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        FFT_spins(spins);

        auto& ft_D_matrices = fft_plan_d.cpx_ptr;
        auto& ft_spins = fft_plan_spins.cpx_ptr;

        auto& res_iFFT = fft_plan_rev.real_ptr;
        auto& res_mult = fft_plan_rev.cpx_ptr;

        field<int> iteration_bounds = { geometry->n_cell_atoms, 
                                        (n_cells_padded[0]/2+1), 
                                        n_cells_padded[1], 
                                        n_cells_padded[2] };

        int number_of_mults = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];

        // Loop over basis atoms (i.e sublattices)
        for(int i_b1 = 0; i_b1 < geometry->n_cell_atoms; ++i_b1)
        {

            CU_FFT_Pointwise_Mult<<<(number_of_mults + 1023) / 1024, 1024>>>(ft_D_matrices.data(), ft_spins.data(), res_mult.data(), iteration_bounds.data(), i_b1, b_diff_lookup.data(), d_stride, spin_stride);
            FFT::batch_iFour_3D(fft_plan_rev);

            //Place the gradients at the correct positions and mult with correct mu

            CU_Write_FFT_Gradients<<<(geometry->n_cells_total + 1023) / 1024, 1024>>>(res_iFFT.data(), gradient.data(), spin_stride, geometry->n_cells.data(), geometry->n_cell_atoms, i_b1, geometry->mu_s.data(), sublattice_size);
        }//end iteration sublattice 1
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

     __global__ void CU_Write_FFT_Input(FFT::FFT_real_type* fft_spin_inputs, const Vector3 * spins, int * iteration_bounds, FFT::StrideContainer spin_stride, scalar * mu_s)
                {
        int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
        int tupel[4];
        int idx_pad;
        for(int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x)
                    {
            tupel_from_idx(idx_orig, tupel, iteration_bounds, 4); //tupel nowis {ib, a, b, c}
            idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
            fft_spin_inputs[idx_pad                        ] = spins[idx_orig][0] * mu_s[idx_orig];
            fft_spin_inputs[idx_pad + 1 * spin_stride.comp ] = spins[idx_orig][1] * mu_s[idx_orig];
            fft_spin_inputs[idx_pad + 2 * spin_stride.comp ] = spins[idx_orig][2] * mu_s[idx_orig];
            }
        }
    void Hamiltonian_Heisenberg::FFT_spins(const vectorfield & spins)
    {
        field<int> iteration_bounds =   {
                                            geometry->n_cell_atoms,
                                            geometry->n_cells[0],
                                            geometry->n_cells[1],
                                            geometry->n_cells[2],
                                        };

        CU_Write_FFT_Input<<<(geometry->nos + 1023) / 1024, 1024>>>(fft_plan_spins.real_ptr.data(), spins.data(), iteration_bounds.data(), spin_stride, geometry->mu_s.data());
        FFT::batch_Four_3D(fft_plan_spins);
    }

    void Hamiltonian_Heisenberg::FFT_Dipole_Mats(int img_a, int img_b, int img_c)
    {
        //prefactor of ddi interaction
        scalar mult = 2 * C::mu_0 * C::mu_B * C::mu_B / ( 4*C::Pi * 1e-30 );

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
                for(int c = 0; c < n_cells_padded[2]; ++c)
                {
                    for(int b = 0; b < n_cells_padded[1]; ++b)
                    {
                        for(int a = 0; a < n_cells_padded[0]; ++a)
                        {
                            int a_idx = a < Na ? a : a - n_cells_padded[0];
                            int b_idx = b < Nb ? b : b - n_cells_padded[1];
                            int c_idx = c < Nc ? c : c - n_cells_padded[2];
                            scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                            Vector3 diff;
                            //iterate over periodic images

                            for(int a_pb = - img_a; a_pb <= img_a; a_pb++)
                            {
                                for(int b_pb = - img_b; b_pb <= img_b; b_pb++)
                                {
                                    for(int c_pb = -img_c; c_pb <= img_c; c_pb++)
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

                            
                        }
                    }
                }
            }
        }
        FFT::batch_Four_3D(fft_plan_d);
    }

    void Hamiltonian_Heisenberg::Prepare_DDI()
    {
        n_cells_padded.resize(3);
        n_cells_padded[0] = (geometry->n_cells[0] > 1) ? 2 * geometry->n_cells[0] : 1;
        n_cells_padded[1] = (geometry->n_cells[1] > 1) ? 2 * geometry->n_cells[1] : 1;
        n_cells_padded[2] = (geometry->n_cells[2] > 1) ? 2 * geometry->n_cells[2] : 1;
        sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

        b_diff_lookup.resize(geometry->n_cell_atoms * geometry->n_cell_atoms);


        //we dont need to transform over length 1 dims
        std::vector<int> fft_dims;
        for(int i = 2; i >= 0; i--) //notice that reverse order is important!
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


        field<int*> temp_s = {&spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c};
        field<int*> temp_d = {&d_stride.comp, &d_stride.basis, &d_stride.a, &d_stride.b, &d_stride.c};;
        FFT::get_strides(temp_s, {3, this->geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2]});
        FFT::get_strides(temp_d, {6, symmetry_count, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2]});
       
        //perform FFT of dipole matrices
        int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
        int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
        int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];
        FFT_Dipole_Mats(img_a, img_b, img_c);

    }//end prepare

    // Hamiltonian name as string
    static const std::string name = "Heisenberg";
    const std::string& Hamiltonian_Heisenberg::Name() { return name; }
}

#endif