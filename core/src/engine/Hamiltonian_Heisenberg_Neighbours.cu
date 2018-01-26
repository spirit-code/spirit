#ifdef USE_CUDA

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>

#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>

using namespace Data;
using namespace Utility;
using Engine::Vectormath::cu_check_atom_type;
using Engine::Vectormath::cu_idx_from_pair;

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
		external_field_magnitude(external_field_magnitude * Constants::mu_B), external_field_normal(external_field_normal),
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
			dmi_normals.push_back(Neighbours::DMI_Normal_from_Pair(*geometry, { dmi_neighbours[ineigh].i, dmi_neighbours[ineigh].j, {dmi_neighbours[ineigh].translations[0], dmi_neighbours[ineigh].translations[1], dmi_neighbours[ineigh].translations[2]} }, dm_chirality));
		}

		// Generate DDI neighbours, magnitudes and normals
		this->ddi_neighbours = Engine::Neighbours::Get_Neighbours_in_Radius(*this->geometry, ddi_radius);
		scalar magnitude;
		Vector3 normal;
		for (unsigned int i=0; i<ddi_neighbours.size(); ++i)
		{
		    Engine::Neighbours::DDI_from_Pair(*this->geometry, {ddi_neighbours[i].i, ddi_neighbours[i].j, {ddi_neighbours[i].translations[0], ddi_neighbours[i].translations[1], ddi_neighbours[i].translations[2]}}, magnitude, normal);
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

		CU_CHECK_ERROR();
		CU_HANDLE_ERROR( cudaDeviceSynchronize() );
	}


	__global__ void HNeigh_CU_E_Zeeman(const Vector3 * spins, const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude, const Vector3 external_field_normal, scalar * Energy, size_t n_cells_total)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n_cells_total;
			idx +=  blockDim.x * gridDim.x)
		{
			for (int ibasis=0; ibasis<n_cell_atoms; ++ibasis)
			{
				int ispin = idx + ibasis;
				if ( cu_check_atom_type(atom_types[ispin]) )
					Energy[ispin] -= mu_s[ibasis] * external_field_magnitude * external_field_normal.dot(spins[ispin]);
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		int size = geometry->n_cells_total;
		HNeigh_CU_E_Zeeman<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), geometry->n_cell_atoms, this->mu_s.data(), this->external_field_magnitude, this->external_field_normal, Energy.data(), size);
	}

	__global__ void HNeigh_CU_E_Anisotropy(const Vector3 * spins, const int * atom_types, const int n_anisotropies, const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, scalar * Energy, size_t n_cells_total)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n_cells_total;
			idx +=  blockDim.x * gridDim.x)
		{
			for (int iani=0; iani<n_anisotropies; ++iani)
			{
				int ispin = idx + anisotropy_indices[iani];
				if ( cu_check_atom_type(atom_types[ispin]) )
					Energy[ispin] -= anisotropy_magnitude[iani] * std::pow(anisotropy_normal[iani].dot(spins[ispin]), 2.0);
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
	{
		int size = geometry->n_cells_total;
		HNeigh_CU_E_Anisotropy<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), this->anisotropy_indices.size(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), Energy.data(), size);
	}

	__global__ void HNeigh_CU_E_Exchange(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_neigh, const Neighbour * neighbours, const scalar * magnitudes, scalar * Energy, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

		for (auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			for (unsigned int ineigh = 0; ineigh < n_neigh; ++ineigh)
			{
				int jspin = cu_idx_from_pair(ispin, bc, nc, n_basis_spins, atom_types, neighbours[ineigh]);
				if ( jspin >= 0 )
				{
					auto& ishell = neighbours[ineigh].idx_shell;
					Energy[ispin] -= 0.5 * magnitudes[ishell] * spins[ispin].dot(spins[jspin]);
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Exchange(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		HNeigh_CU_E_Exchange<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
				this->exchange_neighbours.size(), this->exchange_neighbours.data(), this->exchange_magnitudes.data(), Energy.data(), size );
	}

	__global__ void HNeigh_CU_E_DMI(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_neighbours, const Neighbour * neighbours, const scalar * magnitudes, const Vector3 * normals, scalar * Energy, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			for (unsigned int ineigh = 0; ineigh < n_neighbours; ++ineigh)
			{
				int jspin = cu_idx_from_pair(ispin, bc, nc, n_basis_spins, atom_types, neighbours[ineigh]);
				if ( jspin >= 0 )
				{
					auto& ishell = neighbours[ineigh].idx_shell;
					Energy[ispin] -= 0.5 * magnitudes[ishell] * normals[ineigh].dot(spins[ispin].cross(spins[jspin]));
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_DMI(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		HNeigh_CU_E_DMI<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
				this->dmi_neighbours.size(), this->dmi_neighbours.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), Energy.data(), size );
	}

	void Hamiltonian_Heisenberg_Neighbours::E_DDI(const vectorfield & spins, scalarfield & Energy)
	{
		// //scalar mult = -mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
		// scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		// scalar result = 0.0;

		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
		// 	{
		// 		if (ddi_magnitudes[ineigh] > 0.0)
		// 		{
		// 			auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
		// 			if ( Vectormath::boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, ddi_neighbours[ineigh].translations) )
		// 			{
		// 				int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, ddi_neighbours[ineigh].translations);

		// 				Energy[ispin] -= mult / std::pow(ddi_magnitudes[ineigh], 3.0) *
		// 					(3 * spins[jspin].dot(ddi_normals[ineigh]) * spins[ispin].dot(ddi_normals[ineigh]) - spins[ispin].dot(spins[jspin]));
		// 				Energy[jspin] -= mult / std::pow(ddi_magnitudes[ineigh], 3.0) *
		// 					(3 * spins[jspin].dot(ddi_normals[ineigh]) * spins[ispin].dot(ddi_normals[ineigh]) - spins[ispin].dot(spins[jspin]));
		// 			}
		// 		}
		// 	}
		// }
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

		CU_CHECK_ERROR();
		CU_HANDLE_ERROR( cudaDeviceSynchronize() );
	}

	__global__ void HNeigh_CU_Gradient_Zeeman( const int * atom_types, const int n_cell_atoms, const scalar * mu_s, const scalar external_field_magnitude, const Vector3 external_field_normal, Vector3 * gradient, size_t n_cells_total)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n_cells_total;
			idx +=  blockDim.x * gridDim.x)
		{
			for (int ibasis=0; ibasis<n_cell_atoms; ++ibasis)
			{
				int ispin = idx + ibasis;
				if ( cu_check_atom_type(atom_types[ispin]) )
				{
					gradient[ispin] -= mu_s[ibasis] * external_field_magnitude*external_field_normal;
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Zeeman(vectorfield & gradient)
	{
		int size = geometry->n_cells_total;
		HNeigh_CU_Gradient_Zeeman<<<(size+1023)/1024, 1024>>>( this->geometry->atom_types.data(), geometry->n_cell_atoms, this->mu_s.data(), this->external_field_magnitude, this->external_field_normal, gradient.data(), size );
	}

	__global__ void HNeigh_CU_Gradient_Anisotropy(const Vector3 * spins, const int * atom_types, const int n_anisotropies, const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, Vector3 * gradient, size_t n_cells_total)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < n_cells_total;
			idx +=  blockDim.x * gridDim.x)
		{
			for (int iani=0; iani<n_anisotropies; ++iani)
			{
				int ispin = idx+anisotropy_indices[iani];
				if ( cu_check_atom_type(atom_types[ispin]) )
				{
					scalar sc = -2 * anisotropy_magnitude[iani] * anisotropy_normal[iani].dot(spins[ispin]);
					gradient[ispin] += sc*anisotropy_normal[iani];
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
	{
		int size = geometry->n_cells_total;
		HNeigh_CU_Gradient_Anisotropy<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), this->anisotropy_indices.size(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), gradient.data(), size );
	}

	__global__ void HNeigh_CU_Gradient_Exchange(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_neigh, const Neighbour * neighbours, const scalar * magnitudes, Vector3 * gradient, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

		for (auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			Vector3 grad{0,0,0};
			Vector3 spin=spins[ispin];
			
			for (unsigned int ineigh = 0; ineigh < n_neigh; ++ineigh)
			{
				int jspin = cu_idx_from_pair(ispin, bc, nc, n_basis_spins, atom_types, neighbours[ineigh]);
				if ( jspin >= 0 )
				{
					auto& ishell = neighbours[ineigh].idx_shell;
					grad -= magnitudes[ishell] * spins[jspin];
				}
			}
			gradient[ispin] += grad;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		HNeigh_CU_Gradient_Exchange<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
				this->exchange_neighbours.size(), this->exchange_neighbours.data(), this->exchange_magnitudes.data(), gradient.data(), size );
	}

	__global__ void HNeigh_CU_Gradient_DMI(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_neighbours, const Neighbour * neighbours, const scalar * magnitudes, const Vector3 * normals, Vector3 * gradient, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			Vector3 grad{0,0,0};
			Vector3 spin=spins[ispin]; 
			for (unsigned int ineigh = 0; ineigh < n_neighbours; ++ineigh)
			{
				int jspin = cu_idx_from_pair(ispin, bc, nc, n_basis_spins, atom_types, neighbours[ineigh]);
				if ( jspin >= 0 )
				{
					auto& ishell = neighbours[ineigh].idx_shell;
					grad -= magnitudes[ishell]*spins[jspin].cross(normals[ineigh]);
				}
			}
			gradient[ispin] += grad;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		HNeigh_CU_Gradient_DMI<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_cell_atoms,
				this->dmi_neighbours.size(), this->dmi_neighbours.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), gradient.data(), size );
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
	{
		// //scalar mult = mu_B*mu_B*1.0 / 4.0 / Pi; // multiply with mu_B^2
		// scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		
		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
		// 	{
		// 		if (ddi_magnitudes[ineigh] > 0.0)
		// 		{
		// 			// std::cerr << ineigh << std::endl;
		// 			auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
		// 			if ( Vectormath::boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, ddi_neighbours[ineigh].translations) )
		// 			{
		// 				int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, ddi_neighbours[ineigh].translations);

		// 				if (ddi_magnitudes[ineigh] > 0.0)
		// 				{
		// 					scalar skalar_contrib = mult / std::pow(ddi_magnitudes[ineigh], 3.0);
		// 					gradient[ispin] -= skalar_contrib * (3 * ddi_normals[ineigh] * spins[jspin].dot(ddi_normals[ineigh]) - spins[jspin]);
		// 					gradient[jspin] -= skalar_contrib * (3 * ddi_normals[ineigh] * spins[ispin].dot(ddi_normals[ineigh]) - spins[ispin]);
		// 				}
		// 			}
		// 		}
		// 	}
		// }
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

		// // Spin Pair elements
		// // Exchange
		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
		// 	for (unsigned int ineigh = 0; ineigh < this->exchange_neighbours.size(); ++ineigh)
		// 	{
		// 		for (int alpha = 0; alpha < 3; ++alpha)
		// 		{
		// 			//int idx_i = 3 * exchange_neighbours[i_pair][0] + alpha;
		// 			//int idx_j = 3 * exchange_neighbours[i_pair][1] + alpha;
		// 			int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_cell_atoms, translations, exchange_neighbours[ineigh].translations);
		// 			int ishell = exchange_neighbours[ineigh].idx_shell;
		// 			hessian(ispin, jspin) += -exchange_magnitudes[ineigh];
		// 			hessian(jspin, ispin) += -exchange_magnitudes[ineigh];
		// 		}
		// 	}
		// }
		// // DMI
		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_cell_atoms, ispin);
		// 	for (unsigned int ineigh = 0; ineigh < this->dmi_neighbours.size(); ++ineigh)
		// 	{
		// 		for (int alpha = 0; alpha < 3; ++alpha)
		// 		{
		// 			for (int beta = 0; beta < 3; ++beta)
		// 			{
		// 				int idx_i = 3 * dmi_neighbours[ineigh].i + alpha;
		// 				int idx_j = 3 * dmi_neighbours[ineigh].j + beta;
		// 				if ((alpha == 0 && beta == 1))
		// 				{
		// 					hessian(idx_i, idx_j) +=
		// 						-dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
		// 					hessian(idx_j, idx_i) +=
		// 						-dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
		// 				}
		// 				else if ((alpha == 1 && beta == 0))
		// 				{
		// 					hessian(idx_i, idx_j) +=
		// 						dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
		// 					hessian(idx_j, idx_i) +=
		// 						dmi_magnitudes[ineigh] * dmi_normals[ineigh][2];
		// 				}
		// 				else if ((alpha == 0 && beta == 2))
		// 				{
		// 					hessian(idx_i, idx_j) +=
		// 						dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
		// 					hessian(idx_j, idx_i) +=
		// 						dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
		// 				}
		// 				else if ((alpha == 2 && beta == 0))
		// 				{
		// 					hessian(idx_i, idx_j) +=
		// 						-dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
		// 					hessian(idx_j, idx_i) +=
		// 						-dmi_magnitudes[ineigh] * dmi_normals[ineigh][1];
		// 				}
		// 				else if ((alpha == 1 && beta == 2))
		// 				{
		// 					hessian(idx_i, idx_j) +=
		// 						-dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
		// 					hessian(idx_j, idx_i) +=
		// 						-dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
		// 				}
		// 				else if ((alpha == 2 && beta == 1))
		// 				{
		// 					hessian(idx_i, idx_j) +=
		// 						dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
		// 					hessian(idx_j, idx_i) +=
		// 						dmi_magnitudes[ineigh] * dmi_normals[ineigh][0];
		// 				}
		// 			}
		// 		}
		// 	}
		// }
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