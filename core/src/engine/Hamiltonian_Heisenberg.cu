#ifdef SPIRIT_USE_CUDA

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
using Engine::Vectormath::cu_check_atom_type;
using Engine::Vectormath::cu_idx_from_pair;

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
        Hamiltonian(boundary_conditions),
        geometry(geometry),
        mu_s(mu_s),
        external_field_magnitude(external_field_magnitude * mu_B), external_field_normal(external_field_normal),
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
        scalarfield mu_s,
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
        mu_s(mu_s),
        external_field_magnitude(external_field_magnitude * mu_B), external_field_normal(external_field_normal),
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
                this->dmi_normals.push_back(Neighbours::DMI_Normal_from_Pair(*geometry, dmi_pairs[ineigh], dmi_shell_chirality));
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
                { this->ddi_pairs[i].i, this->ddi_pairs[i].j, {ddi_pairs[i].translations[0], ddi_pairs[i].translations[1], ddi_pairs[i].translations[2]} },
                this->ddi_magnitudes[i], this->ddi_normals[i]);
        }

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
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
        int nos = spins.size();
        for (auto& pair : contributions)
        {
            // Allocate if not already allocated
            if (pair.second.size() != nos) pair.second = scalarfield(nos, 0);
            // Otherwise set to zero
            else for (auto& pair : contributions) Vectormath::fill(pair.second, 0);
        }
        
        // External field
        if (this->idx_zeeman >=0 )     E_Zeeman(spins, contributions[idx_zeeman].second);

        // Anisotropy
        if (this->idx_anisotropy >=0 ) E_Anisotropy(spins, contributions[idx_anisotropy].second);

        // Pairs
        //    Exchange
        if (this->idx_exchange >=0 )   E_Exchange(spins, contributions[idx_exchange].second);
        //    DMI
        if (this->idx_dmi >=0 )        E_DMI(spins, contributions[idx_dmi].second);
        //    DDI
        if (this->idx_ddi >=0 )        E_DDI(spins, contributions[idx_ddi].second);
        
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
                    Energy[ispin] -= mu_s[ibasis] * external_field_magnitude * external_field_normal.dot(spins[ispin]);
            }
        }
    }
    void Hamiltonian_Heisenberg::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
    {
        int size = geometry->n_cells_total;
        CU_E_Zeeman<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), geometry->n_cell_atoms, this->mu_s.data(), this->external_field_magnitude, this->external_field_normal, Energy.data(), size);
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
        // //scalar mult = -mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
        // scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        // // scalar result = 0.0;

        // for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
        // {
        // 	if (ddi_magnitudes[i_pair] > 0.0)
        // 	{
        // 		for (int da = 0; da < geometry->n_cells[0]; ++da)
        // 		{
        // 			for (int db = 0; db < geometry->n_cells[1]; ++db)
        // 			{
        // 				for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
        // 				{
        // 					std::array<int, 3 > translations = { da, db, dc };
        // 					// int idx_i = ddi_pairs[i_pair].i;
        // 					// int idx_j = ddi_pairs[i_pair].j;
        // 					int idx_i = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
        // 					int idx_j = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, ddi_pairs[i_pair].translations);
        // 					Energy[idx_i] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
        // 						(3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) - spins[idx_i].dot(spins[idx_j]));
        // 					Energy[idx_j] -= mult / std::pow(ddi_magnitudes[i_pair], 3.0) *
        // 						(3 * spins[idx_j].dot(ddi_normals[i_pair]) * spins[idx_i].dot(ddi_normals[i_pair]) - spins[idx_i].dot(spins[idx_j]));
        // 				}
        // 			}
        // 		}
        // 	}
        // }
    }// end DipoleDipole


    void Hamiltonian_Heisenberg::E_Quadruplet(const vectorfield & spins, scalarfield & Energy)
    {
        // for (unsigned int iquad = 0; iquad < quadruplets.size(); ++iquad)
        // {
        // 	for (int da = 0; da < geometry->n_cells[0]; ++da)
        // 	{
        // 		for (int db = 0; db < geometry->n_cells[1]; ++db)
        // 		{
        // 			for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
        // 			{
        // 				std::array<int, 3 > translations = { da, db, dc };
        // 				// int i = quadruplets[iquad].i;
        // 				// int j = quadruplets[iquad].j;
        // 				// int k = quadruplets[iquad].k;
        // 				// int l = quadruplets[iquad].l;
        // 				int i = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
        // 				int j = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_j);
        // 				int k = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_k);
        // 				int l = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, quadruplets[iquad].d_l);
        // 				Energy[i] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        // 				Energy[j] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        // 				Energy[k] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        // 				Energy[l] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
        // 			}
        // 		}
        // 	}
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
                Energy -= this->mu_s[ibasis] * this->external_field_magnitude * this->external_field_normal.dot(spins[ispin_in]);
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
                    // The translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
                    const scalar mult = 0.5 * this->mu_s[ddi_pairs[ipair].i] * this->mu_s[ddi_pairs[ipair].j]
                        * Utility::Constants::mu_0 * std::pow(Utility::Constants::mu_B, 2) / ( 4*Utility::Constants::Pi * 1e-30 );

                    int ispin = ddi_pairs[ipair].i + icell*geometry->n_cell_atoms;
                    int jspin = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, ddi_pairs[ipair]);

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
        Gradient_Zeeman(gradient);

        // Anisotropy
        Gradient_Anisotropy(spins, gradient);

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
                    gradient[ispin] -= mu_s[ibasis] * external_field_magnitude*external_field_normal;
            }
        }
    }
    void Hamiltonian_Heisenberg::Gradient_Zeeman(vectorfield & gradient)
    {
        int size = geometry->n_cells_total;
        CU_Gradient_Zeeman<<<(size+1023)/1024, 1024>>>( this->geometry->atom_types.data(), geometry->n_cell_atoms, this->mu_s.data(), this->external_field_magnitude, this->external_field_normal, gradient.data(), size );
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
        // //scalar mult = mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
        // scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
        
        // for (unsigned int i_pair = 0; i_pair < ddi_pairs.size(); ++i_pair)
        // {
        // 	if (ddi_magnitudes[i_pair] > 0.0)
        // 	{
        // 		for (int da = 0; da < geometry->n_cells[0]; ++da)
        // 		{
        // 			for (int db = 0; db < geometry->n_cells[1]; ++db)
        // 			{
        // 				for (int dc = 0; dc < geometry->n_cells[2]; ++dc)
        // 				{
        // 					scalar skalar_contrib = mult / std::pow(ddi_magnitudes[i_pair], 3.0);
        // 					// int idx_i = ddi_pairs[i_pair].i;
        // 					// int idx_j = ddi_pairs[i_pair].j;
        // 					std::array<int, 3 > translations = { da, db, dc };
        // 					if (Vectormath::boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, ddi_pairs[i_pair].translations))
        // 					{
        // 						int ispin = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations);
        // 						int jspin = idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, ddi_pairs[i_pair].translations);
        // 						gradient[ispin] -= skalar_contrib * (3 * ddi_normals[i_pair] * spins[jspin].dot(ddi_normals[i_pair]) - spins[jspin]);
        // 						gradient[jspin] -= skalar_contrib * (3 * ddi_normals[i_pair] * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin]);
        // 					}
        // 				}
        // 			}
        // 		}
        // 	}
        // }
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