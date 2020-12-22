#ifdef SPIRIT_USE_CUDA

#include <engine/Hamiltonian_Micromagnetic.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <engine/FFT.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <algorithm>

using namespace Data;
using namespace Utility;
namespace C = Utility::Constants_Micromagnetic;
using Engine::Vectormath::check_atom_type;
using Engine::Vectormath::idx_from_pair;
using Engine::Vectormath::cu_check_atom_type;
using Engine::Vectormath::cu_idx_from_pair;
using Engine::Vectormath::cu_tupel_from_idx;


namespace Engine
{
	Hamiltonian_Micromagnetic::Hamiltonian_Micromagnetic(
        scalar Ms,
        scalar external_field_magnitude, Vector3 external_field_normal,
        Matrix3 anisotropy_tensor,
        Matrix3 exchange_tensor,
        Matrix3 dmi_tensor,
        DDI_Method ddi_method, intfield ddi_n_periodic_images, scalar ddi_radius,
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
		neigh = pairfield(0);
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

		// Dipole-dipole
		this->Prepare_DDI();

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Micromagnetic::Update_Energy_Contributions()
    {
        this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);
		CU_CHECK_AND_SYNC();
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

	__global__ void CU_E_Zeeman1(const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude, const Vector3 external_field_normal, scalar * Energy, size_t n_cells_total)
	{
		for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
			icell < n_cells_total;
			icell += blockDim.x * gridDim.x)
		{
			for (int ibasis = 0; ibasis < n_cell_atoms; ++ibasis)
			{
				int ispin = icell + ibasis;
				if (cu_check_atom_type(atom_types[ispin]))
					Energy[ispin] -= mu_s[ispin] * external_field_magnitude * external_field_normal.dot(spins[ispin]);
			}
		}
	}
	void Hamiltonian_Micromagnetic::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		int size = geometry->n_cells_total;
		CU_E_Zeeman1 << <(size + 1023) / 1024, 1024 >> > (spins.data(), this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(), this->external_field_magnitude, this->external_field_normal, Energy.data(), size);
		CU_CHECK_AND_SYNC();
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
		this->Spatial_Gradient(spins);
        // External field
        this->Gradient_Zeeman(gradient);

        // Anisotropy
        this->Gradient_Anisotropy(spins, gradient);

        // Exchange
        this->Gradient_Exchange(spins, gradient);

        // DMI
        this->Gradient_DMI(spins, gradient);
		scalar Ms = 1.4e6;
		double energy = 0;
		#pragma omp parallel for reduction(-:energy)
		for (int icell = 0; icell < geometry->n_cells_total; ++icell)
		{
			//energy -= 0.5 *Ms* gradient[icell].dot(spins[icell]);
		}
		//printf("Energy total: %f\n", energy/ geometry->n_cells_total);

    }


	__global__ void CU_Gradient_Zeeman1(const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude, const Vector3 external_field_normal, Vector3 * gradient, size_t n_cells_total)
	{
		for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
			icell < n_cells_total;
			icell += blockDim.x * gridDim.x)
		{
			for (int ibasis = 0; ibasis < n_cell_atoms; ++ibasis)
			{
				int ispin = icell + ibasis;
				if (cu_check_atom_type(atom_types[ispin]))
					gradient[ispin] -= mu_s[ispin] * C::mu_B * external_field_magnitude*external_field_normal;
			}
		}
	}
	void Hamiltonian_Micromagnetic::Gradient_Zeeman(vectorfield & gradient)
	{
		int size = geometry->n_cells_total;
		CU_Gradient_Zeeman1 << <(size + 1023) / 1024, 1024 >> > (this->geometry->atom_types.data(), geometry->n_cell_atoms, geometry->mu_s.data(), this->external_field_magnitude, this->external_field_normal, gradient.data(), size);
		CU_CHECK_AND_SYNC();
	}

	__global__ void CU_Gradient_Anisotropy1(const Vector3 * spins, const int * atom_types, const int n_cell_atoms, Vector3 * gradient, size_t n_cells_total, Matrix3 anisotropy_tensor)
	{
		scalar Ms = 1.4e6;
		Vector3 temp1{ 1,0,0 };
		Vector3 temp2{ 0,1,0 };
		Vector3 temp3{ 0,0,1 };
		for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
			icell < n_cells_total;
			icell += blockDim.x * gridDim.x)
		{
            int ispin = icell;
            gradient[ispin] -= 2.0 * C::mu_B * anisotropy_tensor * spins[ispin] / Ms;
            //gradient[ispin] -= 2.0 * this->anisotropy_magnitudes[iani] / Ms * ((pow(temp2.dot(spins[ispin]),2)+ pow(temp3.dot(spins[ispin]), 2))*(temp1.dot(spins[ispin])*temp1)+ (pow(temp1.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp2.dot(spins[ispin])*temp2)+(pow(temp1.dot(spins[ispin]),2)+ pow(temp2.dot(spins[ispin]), 2))*(temp3.dot(spins[ispin])*temp3));
            //gradient[ispin] += 2.0 * 50000 / Ms * ((pow(temp2.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp1.dot(spins[ispin])*temp1) + (pow(temp1.dot(spins[ispin]), 2) + pow(temp3.dot(spins[ispin]), 2))*(temp2.dot(spins[ispin])*temp2));
		}
	}

	void Hamiltonian_Micromagnetic::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
	{
		int size = geometry->n_cells_total;
		CU_Gradient_Anisotropy1 << <(size + 1023) / 1024, 1024 >> > (spins.data(), this->geometry->atom_types.data(), this->geometry->n_cell_atoms, gradient.data(), size, this->anisotropy_tensor);
		CU_CHECK_AND_SYNC();
	}

	__global__ void CU_Gradient_Exchange1(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_cell_atoms,
		int n_pairs, const Pair * neigh, Vector3 * gradient, size_t size, bool A_is_nondiagonal, Matrix3 exchange_tensor, const scalar * delta, const scalar Ms)
	{
		int bc[3] = { boundary_conditions[0],boundary_conditions[1],boundary_conditions[2] };

		int nc[3] = { n_cells[0],n_cells[1],n_cells[2] };

		for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
			icell < size;
			icell += blockDim.x * gridDim.x)
		{
			// int ispin = icell;//basically id of a cell
			for (unsigned int i = 0; i < 3; ++i)
			{

				int icell_plus  = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, neigh[2*i]);
				int icell_minus = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, neigh[2*i + 1]);

                if( icell_plus >= 0 || icell_minus >= 0 )
                {
                    if( icell_plus == -1 )
                        icell_plus = icell;
                    if( icell_minus == -1 )
                        icell_minus = icell;

                    gradient[icell] -= 2 * C::mu_B * exchange_tensor * (spins[icell_plus] - 2*spins[icell] + spins[icell_minus]) / (Ms*delta[i]*delta[i]);
                }
			}
			/*if (A_is_nondiagonal == true) {
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
				gradient[ispin][0] -= 2 * exchange_tensor(0, 1) / Ms * ((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](0, 1) - spatial_gradient[ispin_left](0, 1)) / 4 / delta[0]);
				gradient[ispin][0] -= 2 * exchange_tensor(1, 0) / Ms * ((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](0, 1) - spatial_gradient[ispin_left](0, 1)) / 4 / delta[0]);
				gradient[ispin][1] -= 2 * exchange_tensor(0, 1) / Ms * ((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](1, 1) - spatial_gradient[ispin_left](1, 1)) / 4 / delta[0]);
				gradient[ispin][1] -= 2 * exchange_tensor(1, 0) / Ms * ((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](1, 1) - spatial_gradient[ispin_left](1, 1)) / 4 / delta[0]);
				gradient[ispin][2] -= 2 * exchange_tensor(0, 1) / Ms * ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](2, 1) - spatial_gradient[ispin_left](2, 1)) / 4 / delta[0]);
				gradient[ispin][2] -= 2 * exchange_tensor(1, 0) / Ms * ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[1] + (spatial_gradient[ispin_right](2, 1) - spatial_gradient[ispin_left](2, 1)) / 4 / delta[0]);

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
				gradient[ispin][0] -= 2 * exchange_tensor(0, 2) / Ms * ((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);
				gradient[ispin][0] -= 2 * exchange_tensor(2, 0) / Ms * ((spatial_gradient[ispin_top](0, 0) - spatial_gradient[ispin_bottom](0, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);
				gradient[ispin][1] -= 2 * exchange_tensor(0, 2) / Ms * ((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]);
				gradient[ispin][1] -= 2 * exchange_tensor(2, 0) / Ms * ((spatial_gradient[ispin_top](1, 0) - spatial_gradient[ispin_bottom](1, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]);
				gradient[ispin][2] -= 2 * exchange_tensor(0, 2) / Ms * ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);
				gradient[ispin][2] -= 2 * exchange_tensor(2, 0) / Ms * ((spatial_gradient[ispin_top](2, 0) - spatial_gradient[ispin_bottom](2, 0)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);

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
				gradient[ispin][0] -= 2 * exchange_tensor(1, 2) / Ms * ((spatial_gradient[ispin_top](0, 1) - spatial_gradient[ispin_bottom](0, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);
				gradient[ispin][0] -= 2 * exchange_tensor(2, 1) / Ms * ((spatial_gradient[ispin_top](0, 1) - spatial_gradient[ispin_bottom](0, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](0, 2) - spatial_gradient[ispin_left](0, 2)) / 4 / delta[0]);
				gradient[ispin][1] -= 2 * exchange_tensor(1, 2) / Ms * ((spatial_gradient[ispin_top](1, 1) - spatial_gradient[ispin_bottom](1, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]);
				gradient[ispin][1] -= 2 * exchange_tensor(2, 1) / Ms * ((spatial_gradient[ispin_top](1, 1) - spatial_gradient[ispin_bottom](1, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](1, 2) - spatial_gradient[ispin_left](1, 2)) / 4 / delta[0]);
				gradient[ispin][2] -= 2 * exchange_tensor(1, 2) / Ms * ((spatial_gradient[ispin_top](2, 1) - spatial_gradient[ispin_bottom](2, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);
				gradient[ispin][2] -= 2 * exchange_tensor(2, 1) / Ms * ((spatial_gradient[ispin_top](2, 1) - spatial_gradient[ispin_bottom](2, 1)) / 4 / delta[2] + (spatial_gradient[ispin_right](2, 2) - spatial_gradient[ispin_left](2, 2)) / 4 / delta[0]);

			}*/

		}
	}
	void Hamiltonian_Micromagnetic::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
	{
		int size       = geometry->n_cells_total;
        scalar * delta = geometry->cell_size.data();
		CU_Gradient_Exchange1 << <(size + 1023) / 1024, 1024 >> > (spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
			this->neigh.size(), this->neigh.data(), gradient.data(), size, A_is_nondiagonal, exchange_tensor, delta, this->Ms );
		CU_CHECK_AND_SYNC();
	}

	__global__ void CU_Spatial_Gradient(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_cell_atoms,
		int n_pairs, const Pair * neigh, Matrix3 * spatial_gradient, size_t size, scalar * delta, scalar Ms)
	{
		/*
		dn1/dr1 dn1/dr2 dn1/dr3
		dn2/dr1 dn2/dr2 dn2/dr3
		dn3/dr1 dn3/dr2 dn3/dr3
		*/
		int bc[3] = { boundary_conditions[0], boundary_conditions[1], boundary_conditions[2] };
		int nc[3] = { n_cells[0], n_cells[1], n_cells[2] };

		for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
			icell < size;
			icell += blockDim.x * gridDim.x)
        {
			for (unsigned int i = 0; i < 3; ++i)
			{
				int icell_plus  = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, neigh[2*i]);
				int icell_minus = cu_idx_from_pair(icell, bc, nc, n_cell_atoms, atom_types, neigh[2*i + 1]);

			    if( icell_plus >= 0 || icell_minus >= 0 )
                {
                    if( icell_plus == -1 )
                        icell_plus = icell;
                    if( icell_minus == -1 )
                        icell_minus = icell;

                    spatial_gradient[icell].col(i) += (spins[icell_plus] - spins[icell_minus]) / (2*delta[i]);
                }

			}
		}
	}

	void Hamiltonian_Micromagnetic::Spatial_Gradient(const vectorfield & spins)
	{
		int size = geometry->n_cells_total;
		CU_Spatial_Gradient << <(size + 1023) / 1024, 1024 >> > (spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
			this->neigh.size(), this->neigh.data(), spatial_gradient.data(), size, geometry->cell_size.data(), this->Ms);
		CU_CHECK_AND_SYNC();
	}

	__global__ void CU_Gradient_DMI1(const Vector3 * spins, Vector3 * gradient, Matrix3 * spatial_gradient, size_t size, Matrix3 dmi_tensor, scalar Ms)
	{
		for (auto icell = blockIdx.x * blockDim.x + threadIdx.x;
			icell < size;
			icell += blockDim.x * gridDim.x)
		{
			for (unsigned int i = 0; i < 3; ++i)
			{
				gradient[icell][0] -= 4 * C::mu_B * (dmi_tensor(1, i) * spatial_gradient[icell](2, i) - 2 * dmi_tensor(2, i) * spatial_gradient[icell](1, i)) / Ms;
				gradient[icell][1] -= 4 * C::mu_B * (dmi_tensor(2, i) * spatial_gradient[icell](0, i) - 2 * dmi_tensor(0, i) * spatial_gradient[icell](2, i)) / Ms;
				gradient[icell][2] -= 4 * C::mu_B * (dmi_tensor(0, i) * spatial_gradient[icell](1, i) - 2 * dmi_tensor(1, i) * spatial_gradient[icell](0, i)) / Ms;
			}
		}
	}
	void Hamiltonian_Micromagnetic::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
	{
		int size = geometry->n_cells_total;
		CU_Gradient_DMI1 << <(size + 1023) / 1024, 1024 >> > (spins.data(), gradient.data(), spatial_gradient.data(), size, dmi_tensor, this->Ms);
		CU_CHECK_AND_SYNC();
	}


	__global__ void CU_FFT_Pointwise_Mult1(FFT::FFT_cpx_type * ft_D_matrices, FFT::FFT_cpx_type * ft_spins, FFT::FFT_cpx_type * res_mult, int* iteration_bounds, int i_b1, int* inter_sublattice_lookup, FFT::StrideContainer dipole_stride, FFT::StrideContainer spin_stride, const scalar Ms)
	{
		int n = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
		int tupel[4];
		int idx_b1, idx_b2, idx_d;

		for (int ispin = blockIdx.x * blockDim.x + threadIdx.x; ispin < n; ispin += blockDim.x * gridDim.x)
		{
			cu_tupel_from_idx(ispin, tupel, iteration_bounds, 4); // tupel now is {i_b2, a, b, c}

			int& b_inter = inter_sublattice_lookup[i_b1 + tupel[0] * iteration_bounds[0]];

			idx_b1 = i_b1 * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
			idx_b2 = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
			idx_d = b_inter * dipole_stride.basis + tupel[1] * dipole_stride.a + tupel[2] * dipole_stride.b + tupel[3] * dipole_stride.c;

			auto& fs_x = ft_spins[idx_b2];
			auto& fs_y = ft_spins[idx_b2 + 1 * spin_stride.comp];
			auto& fs_z = ft_spins[idx_b2 + 2 * spin_stride.comp];

			auto& fD_xx = ft_D_matrices[idx_d];
			auto& fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
			auto& fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
			auto& fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
			auto& fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
			auto& fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

			if (tupel[0] == 0)
			{
				res_mult[idx_b1].x = FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z).x;
				res_mult[idx_b1].y = FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z).y;
				res_mult[idx_b1 + 1 * spin_stride.comp].x = FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z).x;
				res_mult[idx_b1 + 1 * spin_stride.comp].y = FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z).y;
				res_mult[idx_b1 + 2 * spin_stride.comp].x = FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z).x;
				res_mult[idx_b1 + 2 * spin_stride.comp].y = FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z).y;
			}
			else {
				atomicAdd(&res_mult[idx_b1].x, FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z).x);
				atomicAdd(&res_mult[idx_b1].y, FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z).y);
				atomicAdd(&res_mult[idx_b1 + 1 * spin_stride.comp].x, FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z).x);
				atomicAdd(&res_mult[idx_b1 + 1 * spin_stride.comp].y, FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z).y);
				atomicAdd(&res_mult[idx_b1 + 2 * spin_stride.comp].x, FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z).x);
				atomicAdd(&res_mult[idx_b1 + 2 * spin_stride.comp].y, FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z).y);
			}
		}
	}

	__global__ void CU_Write_FFT_Gradients1(const FFT::FFT_real_type * resiFFT, Vector3 * gradient, FFT::StrideContainer spin_stride, int * iteration_bounds, int n_cell_atoms, scalar * mu_s, int sublattice_size, const scalar Ms)
	{
		int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
		int tupel[4];
		int idx_pad;
		for (int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x)
		{

			cu_tupel_from_idx(idx_orig, tupel, iteration_bounds, 4); //tupel now is {ib, a, b, c}
			idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
			//printf("%d %f %f\n", idx_orig, resiFFT[idx_pad],gradient[idx_orig][0]);
			gradient[idx_orig][0] -= C::mu_B * resiFFT[idx_pad]*Ms*1e-7/(sublattice_size);
			gradient[idx_orig][1] -= C::mu_B * resiFFT[idx_pad + 1 * spin_stride.comp]*Ms*1e-7/(sublattice_size);
			gradient[idx_orig][2] -= C::mu_B * resiFFT[idx_pad + 2 * spin_stride.comp]*Ms*1e-7/(sublattice_size);
		}
	}

	void Hamiltonian_Micromagnetic::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
	{
		//this->Gradient_DDI_Direct(spins, gradient);
		this->Gradient_DDI_FFT(spins, gradient);
		/*
		if (this->ddi_method == DDI_Method::FFT)
		{
			printf("sasas");
			this->Gradient_DDI_FFT(spins, gradient);
		}
			else if (this->ddi_method == DDI_Method::Cutoff)
			{
				// TODO: Merge these implementations in the future
				if (this->ddi_cutoff_radius >= 0)
					this->Gradient_DDI_Cutoff(spins, gradient);
				else
					this->Gradient_DDI_Direct(spins, gradient);
			}
*/
	}
	void Hamiltonian_Micromagnetic::Gradient_DDI_Cutoff(const vectorfield & spins, vectorfield & gradient)
		{
			// TODO
		}
	void Hamiltonian_Micromagnetic::Gradient_DDI_FFT(const vectorfield & spins, vectorfield & gradient)
		{
			auto& ft_D_matrices = transformed_dipole_matrices;

			auto& ft_spins = fft_plan_spins.cpx_ptr;

			auto& res_iFFT = fft_plan_reverse.real_ptr;
			auto& res_mult = fft_plan_reverse.cpx_ptr;

			int number_of_mults = it_bounds_pointwise_mult[0] * it_bounds_pointwise_mult[1] * it_bounds_pointwise_mult[2] * it_bounds_pointwise_mult[3];

			FFT_Spins(spins);

			// TODO: also parallelize over i_b1
			// Loop over basis atoms (i.e sublattices) and add contribution of each sublattice
			for (int i_b1 = 0; i_b1 < geometry->n_cell_atoms; ++i_b1)
				CU_FFT_Pointwise_Mult1<<<(number_of_mults+1023)/1024, 1024>>>(ft_D_matrices.data(), ft_spins.data(), res_mult.data(), it_bounds_pointwise_mult.data(), i_b1, inter_sublattice_lookup.data(), dipole_stride, spin_stride, Ms);
				CU_CHECK_AND_SYNC();
			FFT::batch_iFour_3D(fft_plan_reverse);
			// scalar * delta = geometry->cell_size.data();
			int sublattice_size = it_bounds_write_dipole[0] * it_bounds_write_dipole[1] * it_bounds_write_dipole[2];
			CU_Write_FFT_Gradients1<<<(geometry->nos+1023)/1024, 1024>>>(res_iFFT.data(), gradient.data(), spin_stride, it_bounds_write_gradients.data(), geometry->n_cell_atoms, geometry->mu_s.data(), sublattice_size, Ms);
			CU_CHECK_AND_SYNC();
		}//end Field_DipoleDipole

	void Hamiltonian_Micromagnetic::Gradient_DDI_Direct(const vectorfield & spins, vectorfield & gradient)
		{
			int tupel1[3];
			int tupel2[3];
			int sublattice_size = it_bounds_write_dipole[0] * it_bounds_write_dipole[1] * it_bounds_write_dipole[2];
					//prefactor of ddi interaction
					//scalar mult = 2.0133545*1e-28 * 0.057883817555 * 0.057883817555 / (4 * 3.141592653589793238462643383279502884197169399375105820974 * 1e-30);
			scalar mult = 1 / (4 * 3.141592653589793238462643383279502884197169399375105820974);
			scalar m0 = (4 * 3.141592653589793238462643383279502884197169399375105820974)*1e-7;
			int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
			int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
			int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];
			scalar * delta = geometry->cell_size.data();
			for (int idx1 = 0; idx1 < geometry->nos; idx1++)
			{
				double kk=0;
				for (int idx2 = 0; idx2 < geometry->nos; idx2++)
				{
					int a1 = idx1%(it_bounds_write_spins[1]);
					int b1 = ((int)(idx1/it_bounds_write_spins[1]))%(it_bounds_write_spins[2]);
					int c1 = (int)idx1/(it_bounds_write_spins[1]*it_bounds_write_spins[2]);
					int a2 = idx2%(it_bounds_write_spins[1]);
					int b2 = ((int)(idx2/it_bounds_write_spins[1]))%(it_bounds_write_spins[2]);
					int c2 = (int)idx2/(it_bounds_write_spins[1]*it_bounds_write_spins[2]);
					/*int a_idx = a < n_cells[0] ? a : a - iteration_bounds[0];
					int b_idx = b < n_cells[1] ? b : b - iteration_bounds[1];
					int c_idx = c < n_cells[2] ? c : c - iteration_bounds[2];*/
					int a_idx = a1-a2;
					int b_idx = b1-b2;
					int c_idx = c1-c2;
					if ((a_idx==0) && (b_idx==0) && (c_idx==0)){
						//printf("test\n");
						//continue;
					}
					//printf("%d %d %d\n", a_idx,b_idx,c_idx);
					/*if ((a_idx==20) || (b_idx==20) || (c_idx==1)){
						//printf("test1\n");
						//if (c_idx!=1)
							//printf("%d %d %d %d\n", a_idx, b_idx, c_idx,  dipole_stride.comp);
						continue;
					}*/
					//scalar delta[3] = { 3,3,0.3 };
					//int idx = b_inter * dipole_stride.basis + a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;
					scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;

					Vector3 cell_sizes = {geometry->lattice_constant * geometry->bravais_vectors[0].norm(),
											geometry->lattice_constant * geometry->bravais_vectors[1].norm(),
											geometry->lattice_constant * geometry->bravais_vectors[2].norm()};
					//asa
					for (int i = 0; i < 2; i++) {
						for (int j = 0; j < 2; j++) {
							for (int k = 0; k < 2; k++) {
								double r = sqrt((a_idx + i - 0.5f)*(a_idx + i - 0.5f)*cell_sizes[0]* cell_sizes[0] + (b_idx + j - 0.5f)*(b_idx + j-0.5f)*cell_sizes[1] * cell_sizes[1] + (c_idx + k - 0.5f)*(c_idx + k - 0.5f)*cell_sizes[2] * cell_sizes[2]);
								Dxx += mult * pow(-1.0f, i + j + k) * atan(((c_idx + k-0.5f) * (b_idx + j - 0.5f) * cell_sizes[1]*cell_sizes[2]/cell_sizes[0] / r / (a_idx + i - 0.5f)));
								//fft_dipole_inputs[idx + 1 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((c_idx + k - 0.5f)* cell_sizes[2] + r)/((c_idx + k - 0.5f)* cell_sizes[2] - r)));
								//fft_dipole_inputs[idx + 2 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((b_idx + j - 0.5f)* cell_sizes[1] + r)/((b_idx + j - 0.5f)* cell_sizes[1] - r)));
								Dxy -= mult * pow(-1.0f, i + j + k) * log((((c_idx + k - 0.5f)* cell_sizes[2] + r)));
								Dxz -= mult * pow(-1.0f, i + j + k) * log((((b_idx + j - 0.5f)* cell_sizes[1] + r)));

								Dyy += mult * pow(-1.0f, i + j + k) * atan(((a_idx + i-0.5f) * (c_idx + k - 0.5f) * cell_sizes[2]*cell_sizes[0]/cell_sizes[1] / r / (b_idx + j - 0.5f)));
								//fft_dipole_inputs[idx + 4 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((a_idx + i - 0.5f)* cell_sizes[0] + r)/((a_idx + i - 0.5f)* cell_sizes[0] - r)));
								Dyz -= mult * pow(-1.0f, i + j + k) * log((((a_idx + i - 0.5f)* cell_sizes[0] + r)));
								Dzz += mult * pow(-1.0f, i + j + k) * atan(((b_idx + j-0.5f) * (a_idx + i - 0.5f) * cell_sizes[0]*cell_sizes[1]/cell_sizes[2] / r / (c_idx + k - 0.5f)));

							}
						}
					}/*
					Dxx=Nii(a_idx*delta[0],b_idx*delta[1],c_idx*delta[2],delta[0],delta[1],delta[2]);
					Dxy=Nij(a_idx*delta[0],b_idx*delta[1],c_idx*delta[2],delta[0],delta[1],delta[2]);
					Dxz=Nij(a_idx*delta[0],c_idx*delta[2], b_idx*delta[1],delta[0],delta[2],delta[1]);
					Dyy=Nii(b_idx*delta[1],a_idx*delta[0],c_idx*delta[2],delta[1],delta[0],delta[2]);
					Dyz=Nij(b_idx*delta[1],c_idx*delta[2], b_idx*delta[1],delta[1],delta[2],delta[0]);
					Dzz=Nii(c_idx*delta[2],a_idx*delta[0],b_idx*delta[1],delta[2],delta[0],delta[1]);*/
					if (idx1==42){
						if ((a_idx==0) && (b_idx==0) && (c_idx==0)){
							printf("000 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==1) && (b_idx==0) && (c_idx==0)){
							printf("100 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==0) && (b_idx==1) && (c_idx==0)){
							printf("010 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==-1) && (b_idx==1) && (c_idx==0)){
							printf("-110 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==1) && (b_idx==1) && (c_idx==0)){
							printf("110 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==2) && (b_idx==0) && (c_idx==0)){
							printf("200 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==0) && (b_idx==2) && (c_idx==0)){
							printf("020 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
						}
						if ((a_idx==2) && (b_idx==2) && (c_idx==0)){
													printf("220 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
												}
						if ((a_idx==2) && (b_idx==-2) && (c_idx==0)){
													printf("2-20 Dxx=%f Dxy=%f Dxz=%f Dyy=%f Dyz=%f Dzz=%f\n",Dxx,Dxy,Dxz,Dyy,Dyz, Dzz);
												}
						//printf("x=%f y=%f z=%f\n",spins[idx2][0],spins[idx2][1],spins[idx2][2]);
					}
					kk+=Dxx;
					gradient[idx1][0] -= (Dxx * spins[idx2][0] + Dxy * spins[idx2][1] + Dxz * spins[idx2][2]) * Ms*m0;
					gradient[idx1][1] -= (Dxy * spins[idx2][0] + Dyy * spins[idx2][1] + Dyz * spins[idx2][2]) * Ms*m0;
					gradient[idx1][2] -= (Dxz * spins[idx2][0] + Dyz * spins[idx2][1] + Dzz * spins[idx2][2]) * Ms*m0;
				}
				if (idx1==30){
					//printf("x=%f y=%f z=%f\n",spins[idx1][0],spins[idx1][1],spins[idx1][2]);
					//printf("kk=%f gx=%f gy=%f gz=%f\n",kk, gradient[idx1][0]/8e5/m0,gradient[idx1][1],gradient[idx1][2]);

				}

			}
		}
	__global__ void CU_Write_FFT_Spin_Input1(FFT::FFT_real_type* fft_spin_inputs, const Vector3 * spins, int * iteration_bounds, FFT::StrideContainer spin_stride, scalar * mu_s)
	{
		int nos = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2] * iteration_bounds[3];
		int tupel[4];
		int idx_pad;
		for (int idx_orig = blockIdx.x * blockDim.x + threadIdx.x; idx_orig < nos; idx_orig += blockDim.x * gridDim.x)
		{
			cu_tupel_from_idx(idx_orig, tupel, iteration_bounds, 4); //tupel now is {ib, a, b, c}
			idx_pad = tupel[0] * spin_stride.basis + tupel[1] * spin_stride.a + tupel[2] * spin_stride.b + tupel[3] * spin_stride.c;
			fft_spin_inputs[idx_pad] = spins[idx_orig][0];
			fft_spin_inputs[idx_pad + 1 * spin_stride.comp] = spins[idx_orig][1];
			fft_spin_inputs[idx_pad + 2 * spin_stride.comp] = spins[idx_orig][2];
			//printf("%f %f\n",fft_spin_inputs[idx_pad], fft_spin_inputs[idx_pad+30]);
		}
	}

	void Hamiltonian_Micromagnetic::FFT_Spins(const vectorfield & spins)
	{
		CU_Write_FFT_Spin_Input1 << <(geometry->nos + 1023) / 1024, 1024 >> > (fft_plan_spins.real_ptr.data(), spins.data(), it_bounds_write_spins.data(), spin_stride, geometry->mu_s.data());
		CU_CHECK_AND_SYNC();
		FFT::batch_Four_3D(fft_plan_spins);
	}
	__global__ void CU_Write_FFT_Dipole_Input1(FFT::FFT_real_type* fft_dipole_inputs, int* iteration_bounds, const Vector3* translation_vectors, int n_cell_atoms, Vector3* cell_atom_translations, int* n_cells, int* inter_sublattice_lookup, int* img, FFT::StrideContainer dipole_stride, const Vector3 cell_lengths)
	{
		int tupel[3];
		int sublattice_size = iteration_bounds[0] * iteration_bounds[1] * iteration_bounds[2];
		//prefactor of ddi interaction
		//scalar mult = 2.0133545*1e-28 * 0.057883817555 * 0.057883817555 / (4 * 3.141592653589793238462643383279502884197169399375105820974 * 1e-30);
		//scalar mult = 1 / (4 * 3.141592653589793238462643383279502884197169399375105820974);
		scalar mult = 1;
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sublattice_size; i += blockDim.x * gridDim.x)
		{
			cu_tupel_from_idx(i, tupel, iteration_bounds, 3); // tupel now is {a, b, c}
			auto& a = tupel[0];
			auto& b = tupel[1];
			auto& c = tupel[2];
			/*if ((a>198)||(b>198)||(c>198)){
				printf("%d %d %d\n", a,b,c);
			}*/
			/*int a_idx = a < n_cells[0] ? a : a - iteration_bounds[0];
			int b_idx = b < n_cells[1] ? b : b - iteration_bounds[1];
			int c_idx = c < n_cells[2] ? c : c - iteration_bounds[2];*/
			/*int a_idx = a +1 - (int)iteration_bounds[0]/2;
			int b_idx = b +1- (int)iteration_bounds[1]/2;
			int c_idx = c +1- (int)iteration_bounds[2]/2;*/
			int a_idx = a < n_cells[0] ? a : a - iteration_bounds[0];
			int b_idx = b < n_cells[1] ? b : b - iteration_bounds[1];
			int c_idx = c < n_cells[2] ? c : c - iteration_bounds[2];

			int idx = a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					for (int k = 0; k < 2; k++) {
						double r = sqrt((a_idx + i - 0.5f)*(a_idx + i - 0.5f)*cell_lengths[0]* cell_lengths[0] + (b_idx + j - 0.5f)*(b_idx + j-0.5f)*cell_lengths[1] * cell_lengths[1] + (c_idx + k - 0.5f)*(c_idx + k - 0.5f)*cell_lengths[2] * cell_lengths[2]);
						fft_dipole_inputs[idx] += mult * pow(-1.0f, i + j + k) * atan(((c_idx + k-0.5f) * (b_idx + j - 0.5f) * cell_lengths[1]*cell_lengths[2]/cell_lengths[0] / r / (a_idx + i - 0.5f)));
						//fft_dipole_inputs[idx + 1 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((c_idx + k - 0.5f)* cell_lengths[2] + r)/((c_idx + k - 0.5f)* cell_lengths[2] - r)));
						//fft_dipole_inputs[idx + 2 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((b_idx + j - 0.5f)* cell_lengths[1] + r)/((b_idx + j - 0.5f)* cell_lengths[1] - r)));
						fft_dipole_inputs[idx + 1 * dipole_stride.comp] -= mult * pow(-1.0f, i + j + k) * log((((c_idx + k - 0.5f)* cell_lengths[2] + r)));
						fft_dipole_inputs[idx + 2 * dipole_stride.comp] -= mult * pow(-1.0f, i + j + k) * log((((b_idx + j - 0.5f)* cell_lengths[1] + r)));

						fft_dipole_inputs[idx + 3 * dipole_stride.comp] += mult * pow(-1.0f, i + j + k) * atan(((a_idx + i-0.5f) * (c_idx + k - 0.5f) * cell_lengths[2]*cell_lengths[0]/cell_lengths[1] / r / (b_idx + j - 0.5f)));
						//fft_dipole_inputs[idx + 4 * dipole_stride.comp] += -mult * pow(-1.0f, i + j + k) * log(abs(((a_idx + i - 0.5f)* cell_lengths[0] + r)/((a_idx + i - 0.5f)* cell_lengths[0] - r)));
						fft_dipole_inputs[idx + 4 * dipole_stride.comp] -= mult * pow(-1.0f, i + j + k) * log((((a_idx + i - 0.5f)* cell_lengths[0] + r)));
						fft_dipole_inputs[idx + 5 * dipole_stride.comp] += mult * pow(-1.0f, i + j + k) * atan(((b_idx + j-0.5f) * (a_idx + i - 0.5f) * cell_lengths[0]*cell_lengths[1]/cell_lengths[2] / r / (c_idx + k - 0.5f)));

					}
				}
			}

				//if (fft_dipole_inputs[idx]<-0.03)
		}
	}

	void Hamiltonian_Micromagnetic::FFT_Dipole_Matrices(FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c)
	{
		auto& fft_dipole_inputs = fft_plan_dipole.real_ptr;

		field<int> img = {
							img_a,
							img_b,
							img_c
		};

		// Work around to make bravais vectors and cell_atoms available to GPU as they are currently saves as std::vectors and not fields ...
		auto translation_vectors = field<Vector3>();
		auto cell_atom_translations = field<Vector3>();

		for (int i = 0; i < 3; i++)
			translation_vectors.push_back(geometry->lattice_constant * geometry->bravais_vectors[i]);

		for (int i = 0; i < geometry->n_cell_atoms; i++)
			cell_atom_translations.push_back(geometry->positions[i]);

		Vector3 cell_sizes = {geometry->lattice_constant * geometry->bravais_vectors[0].norm(),
								geometry->lattice_constant * geometry->bravais_vectors[1].norm(),
								geometry->lattice_constant * geometry->bravais_vectors[2].norm()};

		CU_Write_FFT_Dipole_Input1 << <(sublattice_size + 1023) / 1024, 1024 >> >
			(fft_dipole_inputs.data(), it_bounds_write_dipole.data(), translation_vectors.data(),
				geometry->n_cell_atoms, cell_atom_translations.data(), geometry->n_cells.data(),
				inter_sublattice_lookup.data(), img.data(), dipole_stride, cell_sizes
				);
		CU_CHECK_AND_SYNC();
		FFT::batch_Four_3D(fft_plan_dipole);
	}
	void Hamiltonian_Micromagnetic::Prepare_DDI()
	{
		Clean_DDI();

		n_cells_padded.resize(3);
		n_cells_padded[0] = (geometry->n_cells[0] > 1) ? 2 * geometry->n_cells[0] : 1;
		n_cells_padded[1] = (geometry->n_cells[1] > 1) ? 2 * geometry->n_cells[1] : 1;
		n_cells_padded[2] = (geometry->n_cells[2] > 1) ? 2 * geometry->n_cells[2] : 1;
		sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];
		//printf("111 %d %d %d\n", n_cells_padded[0],n_cells_padded[1],n_cells_padded[2]);

		inter_sublattice_lookup.resize(geometry->n_cell_atoms * geometry->n_cell_atoms);

		//we dont need to transform over length 1 dims
		std::vector<int> fft_dims;
		for (int i = 2; i >= 0; i--) //notice that reverse order is important!
		{
			if (n_cells_padded[i] > 1)
				fft_dims.push_back(n_cells_padded[i]);
		}

		//Count how many distinct inter-lattice contributions we need to store
		n_inter_sublattice = 0;
		for (int i = 0; i < geometry->n_cell_atoms; i++)
		{
			for (int j = 0; j < geometry->n_cell_atoms; j++)
			{
				if (i != 0 && i == j) continue;
				n_inter_sublattice++;
			}
		}
		printf("lex%d %d %d\n", n_inter_sublattice, fft_dims[0],fft_dims[1]);
		//Set the iteration bounds for the nested for loops that are flattened in the kernels
		it_bounds_write_spins = { geometry->n_cell_atoms,
									  geometry->n_cells[0],
									  geometry->n_cells[1],
									  geometry->n_cells[2] };

		it_bounds_write_dipole = { n_cells_padded[0],
									  n_cells_padded[1],
									  n_cells_padded[2] };

		it_bounds_pointwise_mult = { geometry->n_cell_atoms,
									  (n_cells_padded[0] / 2 + 1), // due to redundancy in real fft
									  n_cells_padded[1],
									  n_cells_padded[2] };

		it_bounds_write_gradients = { geometry->n_cell_atoms,
									  geometry->n_cells[0],
									  geometry->n_cells[1],
									  geometry->n_cells[2] };

		FFT::FFT_Plan fft_plan_dipole = FFT::FFT_Plan(fft_dims, false, 6 * n_inter_sublattice, sublattice_size);
		fft_plan_spins = FFT::FFT_Plan(fft_dims, false, 3 * geometry->n_cell_atoms, sublattice_size);
		fft_plan_reverse = FFT::FFT_Plan(fft_dims, true, 3 * geometry->n_cell_atoms, sublattice_size);

		field<int*> temp_s = { &spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c };
		field<int*> temp_d = { &dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c };;
		FFT::get_strides(temp_s, { 3, this->geometry->n_cell_atoms, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] });
		FFT::get_strides(temp_d, { 6, n_inter_sublattice, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2] });
		/*
		//perform FFT of dipole matrices
		int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
		int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
		int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

		FFT_Dipole_Matrices(fft_plan_dipole, img_a, img_b, img_c); */
		FFT_Dipole_Matrices(fft_plan_dipole, 0, 0, 0);

		transformed_dipole_matrices = std::move(fft_plan_dipole.cpx_ptr);
	}//end prepare

	void Hamiltonian_Micromagnetic::Clean_DDI()
	{
		fft_plan_spins = FFT::FFT_Plan();
		fft_plan_reverse = FFT::FFT_Plan();
	}

    void Hamiltonian_Micromagnetic::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
    }


    // Hamiltonian name as string
    static const std::string name = "Micromagnetic";
    const std::string& Hamiltonian_Micromagnetic::Name() { return name; }
}

#endif