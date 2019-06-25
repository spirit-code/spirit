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
namespace C = Utility::Constants_Micromagnetic;
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
        anisotropy_tensor(anisotropy_tensor), exchange_tensor(exchange_tensor), dmi_tensor(dmi_tensor)
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
        anisotropy_tensor(anisotropy_tensor), exchange_tensor(exchange_tensor), dmi_tensor(dmi_tensor)
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

		neigh = neighbourfield(0);
		Neighbour neigh_tmp;
		neigh_tmp.i = 0;
		neigh_tmp.j = 0;
		neigh_tmp.idx_shell = 0;
		//order x -x y -y z -z xy (-x)(-y) x(-y) (-x)y xz (-x)(-z) x(-z) (-x)z yz (-y)(-z) y(-z) (-y)z results in 9 parts of Hessian 
		neigh_tmp.translations[0] = 1;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = -1;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = 1;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = -1;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = 1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = -1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 1;
		neigh_tmp.translations[1] = 1;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = -1;
		neigh_tmp.translations[1] = -1;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 1;
		neigh_tmp.translations[1] = -1;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = -1;
		neigh_tmp.translations[1] = +1;
		neigh_tmp.translations[2] = 0;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 1;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = 1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = -1;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = -1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 1;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = -1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = -1;
		neigh_tmp.translations[1] = 0;
		neigh_tmp.translations[2] = 1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = 1;
		neigh_tmp.translations[2] = 1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = -1;
		neigh_tmp.translations[2] = -1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = 1;
		neigh_tmp.translations[2] = -1;
		neigh.push_back(neigh_tmp);

		neigh_tmp.translations[0] = 0;
		neigh_tmp.translations[1] = -1;
		neigh_tmp.translations[2] = 1;
		neigh.push_back(neigh_tmp);

		this->spatial_gradient = field<Matrix3>(geometry->nos, Matrix3::Zero());
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

	void Hamiltonian_Micromagnetic::E_Update(const vectorfield & spins, scalarfield & Energy, vectorfield & gradient) {
		scalar Ms = 1711e3;
		#pragma omp parallel for
		for (int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			Energy[icell] -= 0.5 *Ms* gradient[icell].dot(spins[icell]);
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
		// std::cout << this->boundary_conditions[0] << std::endl;
		// std::cout << boundary_conditions[0] << std::endl;

        // Set to zero
        Vectormath::fill(gradient, {0,0,0});
		this->Spatial_Gradient(spins);

        // External field
        this->Gradient_Zeeman(gradient);

        // Anisotropy
        this->Gradient_Anisotropy(spins, gradient);

        // Exchange
        this->Gradient_Exchange(spins, gradient);

        // DMI
        this->Gradient_DMI(spins, gradient);
		scalar Ms = 1711e3;
		double energy=0;
		#pragma omp parallel for reduction(-:energy)
		for (int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			energy -= 0.5 *Ms* gradient[icell].dot(spins[icell]);
		}
		// printf("Energy total: %f\n", energy/ geometry->n_cells_total);

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
		scalar Ms = 1711e3;
		Vector3 temp1{ 1,0,0 };
		Vector3 temp2{ 0,1,0 };
		Vector3 temp3{ 0,0,1 };
		#pragma omp parallel for
		for (int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			for (int iani = 0; iani < 1; ++iani)
			{
				int ispin = icell;
				gradient[ispin] += 2.0 * 5e3 / Ms * temp3 * temp3.dot(spins[ispin]);
				//gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[iani] / Ms * ((pow(temp2.dot(spins[ispin]),2)+ pow(temp3.dot(spins[ispin]), 2))*(temp1.dot(spins[ispin])*temp1)+ (pow(temp1.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp2.dot(spins[ispin])*temp2)+(pow(temp1.dot(spins[ispin]),2)+ pow(temp2.dot(spins[ispin]), 2))*(temp3.dot(spins[ispin])*temp3));
				gradient[ispin] += 2.0 * 50000 / Ms * ((pow(temp2.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp1.dot(spins[ispin])*temp1) + (pow(temp1.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp2.dot(spins[ispin])*temp2));

			}
		}
    }

    void Hamiltonian_Micromagnetic::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
		scalar delta[3] = { 4e-8,4e-8,4e-8 };
		scalar Ms = 1711e3;
		//nongradient implementation
		/*
		#pragma omp parallel for
		for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			int ispin = icell;//basically id of a cell
			for (unsigned int i = 0; i < 3; ++i)
			{

				int ispin_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[i]);
				int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[i + 1]);
				if (ispin_plus == -1) {
					ispin_plus = ispin;
				}
				if (ispin_minus == -1) {
					ispin_minus = ispin;
				}
				gradient[ispin][i] -= exchange_tensor(i, i)*(spins[ispin_plus][i] - 2 * spins[ispin][i] + spins[ispin_minus][i]) / (delta[i]) / (delta[i]);

			}
			if (A_is_nondiagonal == true) {
				int ispin_plus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[6]);
				int ispin_minus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[7]);
				int ispin_plus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[8]);
				int ispin_minus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[9]);

				if (ispin_plus_plus == -1) {
					ispin_plus_plus = ispin;
				}
				if (ispin_minus_minus == -1) {
					ispin_minus_minus = ispin;
				}
				if (ispin_plus_minus == -1) {
					ispin_plus_minus = ispin;
				}
				if (ispin_minus_plus == -1) {
					ispin_minus_plus = ispin;
				}
				gradient[ispin][0] -= exchange_tensor(0, 1)*(spins[ispin_plus_plus][1] - spins[ispin_plus_minus][1] - spins[ispin_minus_plus][1] + spins[ispin_minus_minus][1]) / (delta[0]) / (delta[1]) / 4;
				gradient[ispin][1] -= exchange_tensor(1, 0)*(spins[ispin_plus_plus][0] - spins[ispin_plus_minus][0] - spins[ispin_minus_plus][0] + spins[ispin_minus_minus][0]) / (delta[0]) / (delta[1]) / 4;

				ispin_plus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[10]);
				ispin_minus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[11]);
				ispin_plus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[12]);
				ispin_minus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[13]);

				if (ispin_plus_plus == -1) {
					ispin_plus_plus = ispin;
				}
				if (ispin_minus_minus == -1) {
					ispin_minus_minus = ispin;
				}
				if (ispin_plus_minus == -1) {
					ispin_plus_minus = ispin;
				}
				if (ispin_minus_plus == -1) {
					ispin_minus_plus = ispin;
				}
				gradient[ispin][0] -= exchange_tensor(0, 2)*(spins[ispin_plus_plus][2] - spins[ispin_plus_minus][2] - spins[ispin_minus_plus][2] + spins[ispin_minus_minus][2]) / (delta[0]) / (delta[2]) / 4;
				gradient[ispin][2] -= exchange_tensor(2, 0)*(spins[ispin_plus_plus][0] - spins[ispin_plus_minus][0] - spins[ispin_minus_plus][0] + spins[ispin_minus_minus][0]) / (delta[0]) / (delta[2]) / 4;

				ispin_plus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[14]);
				ispin_minus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[15]);
				ispin_plus_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[16]);
				ispin_minus_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[17]);

				if (ispin_plus_plus == -1) {
					ispin_plus_plus = ispin;
				}
				if (ispin_minus_minus == -1) {
					ispin_minus_minus = ispin;
				}
				if (ispin_plus_minus == -1) {
					ispin_plus_minus = ispin;
				}
				if (ispin_minus_plus == -1) {
					ispin_minus_plus = ispin;
				}
				gradient[ispin][1] -= exchange_tensor(1, 2)*(spins[ispin_plus_plus][2] - spins[ispin_plus_minus][2] - spins[ispin_minus_plus][2] + spins[ispin_minus_minus][2]) / (delta[1]) / (delta[2]) / 4;
				gradient[ispin][2] -= exchange_tensor(2, 1)*(spins[ispin_plus_plus][1] - spins[ispin_plus_minus][1] - spins[ispin_minus_plus][1] + spins[ispin_minus_minus][1]) / (delta[1]) / (delta[2]) / 4;
			}

		}*/

		//gradient implementation
		#pragma omp parallel for
		for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			int ispin = icell;//basically id of a cell
			for (unsigned int i = 0; i < 3; ++i)
			{

				int ispin_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[i]);
				int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[i + 1]);
				if (ispin_plus == -1) {
					ispin_plus = ispin;
				}
				if (ispin_minus == -1) {
					ispin_minus = ispin;
				}

				gradient[ispin][i] += exchange_tensor(i, i)/Ms*(spins[ispin_plus][i] - 2 * spins[ispin][i] + spins[ispin_minus][i]) / (delta[i]) / (delta[i]);

			}
			if (this->A_is_nondiagonal == true) {
				//xy
				int ispin_right = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[0]);
				int ispin_left = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[1]);
				int ispin_top = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[2]);
				int ispin_bottom = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[3]);

				if (ispin_right == -1) {
					ispin_right = ispin;
				}
				if (ispin_left == -1) {
					ispin_left = ispin;
				}
				if (ispin_top == -1) {
					ispin_top = ispin;
				}
				if (ispin_bottom == -1) {
					ispin_bottom = ispin;
				}
				gradient[ispin][0] += exchange_tensor(0, 1) / Ms *((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](1, 1) - spatial_gradient[ispin_left](1, 1)) / 4 / delta[0]);
				gradient[ispin][1] += exchange_tensor(1, 0) / Ms *((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](0, 1) - spatial_gradient[ispin_left](0, 1)) / 4 / delta[0]);
				
				//xz
				ispin_right = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[0]);
				ispin_left = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[1]);
				ispin_top = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[4]);
				ispin_bottom = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[5]);

				if (ispin_right == -1) {
					ispin_right = ispin;
				}
				if (ispin_left == -1) {
					ispin_left = ispin;
				}
				if (ispin_top == -1) {
					ispin_top = ispin;
				}
				if (ispin_bottom == -1) {
					ispin_bottom = ispin;
				}
				gradient[ispin][0] += exchange_tensor(0, 2) / Ms *((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);
				gradient[ispin][2] += exchange_tensor(2, 0) / Ms *((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);

				//yz
				ispin_right = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[2]);
				ispin_left = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[3]);
				ispin_top = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[4]);
				ispin_bottom = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[5]);

				if (ispin_right == -1) {
					ispin_right = ispin;
				}
				if (ispin_left == -1) {
					ispin_left = ispin;
				}
				if (ispin_top == -1) {
					ispin_top = ispin;
				}
				if (ispin_bottom == -1) {
					ispin_bottom = ispin;
				}
				gradient[ispin][1] += exchange_tensor(1, 2) / Ms *((spatial_gradient[ispin_top](2, 1) - spatial_gradient[ispin_bottom](2, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[1]);
				gradient[ispin][2] += exchange_tensor(2, 1) / Ms *((spatial_gradient[ispin_top](1, 1) - spatial_gradient[ispin_bottom](1, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 / delta[1]);

			}

		}
		
    }

	void Hamiltonian_Micromagnetic::Spatial_Gradient(const vectorfield & spins) {
		scalar delta[3] = { 4e-8,4e-8,4e-8 };
		/*
		dn1/dr1 dn1/dr2 dn1/dr3
		dn2/dr1 dn2/dr2 dn2/dr3
		dn3/dr1 dn3/dr2 dn3/dr3
		*/
		#pragma omp parallel for
		for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			int ispin = icell;//basically id of a cell
			for (unsigned int i = 0; i < 3; ++i)
			{
				int ispin_plus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[i]);
				int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms, geometry->atom_types, neigh[i + 1]);
				if (ispin_plus == -1) {
					ispin_plus = ispin;
				}
				if (ispin_minus == -1) {
					ispin_minus = ispin;
				}
				spatial_gradient[ispin](0, i) = (spins[ispin_plus][0] - spins[ispin_minus][0]) / (delta[i]) / 2;
				spatial_gradient[ispin](1, i) = (spins[ispin_plus][1] - spins[ispin_minus][1]) / (delta[i]) / 2;
				spatial_gradient[ispin](2, i) = (spins[ispin_plus][2] - spins[ispin_minus][2]) / (delta[i]) / 2;

			}
		}
	}

    void Hamiltonian_Micromagnetic::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
		#pragma omp parallel for
		for (unsigned int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			int ispin = icell;//basically id of a cell
			for (unsigned int i = 0; i < 3; ++i)
			{
				gradient[ispin][0] += 4*dmi_tensor(1, i)*spatial_gradient[ispin](2, i) - 4*dmi_tensor(2, i)*spatial_gradient[ispin](1, i);
				gradient[ispin][1] += 4*dmi_tensor(2, i)*spatial_gradient[ispin](0, i) - 4*dmi_tensor(0, i)*spatial_gradient[ispin](2, i);
				gradient[ispin][2] += 4*dmi_tensor(0, i)*spatial_gradient[ispin](1, i) - 4*dmi_tensor(1, i)*spatial_gradient[ispin](0, i);
			}
		}
    }


    void Hamiltonian_Micromagnetic::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
    }


    // Hamiltonian name as string
    static const std::string name = "Micromagnetic";
    const std::string& Hamiltonian_Micromagnetic::Name() { return name; }
}

#endif