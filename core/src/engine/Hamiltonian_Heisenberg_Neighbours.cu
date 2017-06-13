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

namespace Engine
{
	Hamiltonian_Heisenberg_Neighbours::Hamiltonian_Heisenberg_Neighbours(
		scalarfield mu_s,
		intfield external_field_indices, scalarfield external_field_magnitudes, vectorfield external_field_normals,
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
		external_field_indices(external_field_indices), external_field_magnitudes(external_field_magnitudes), external_field_normals(external_field_normals),
		anisotropy_indices(anisotropy_indices), anisotropy_magnitudes(anisotropy_magnitudes), anisotropy_normals(anisotropy_normals),
		exchange_magnitudes(exchange_magnitudes),
		dmi_magnitudes(dmi_magnitudes),
		ddi_radius(ddi_radius)
	{
		// Atom types
		this->atom_types = intfield(geometry->nos, 0);

		// Renormalize the external field from Tesla to meV
		for (unsigned int i = 0; i < external_field_magnitudes.size(); ++i)
		{
			this->external_field_magnitudes[i] = this->external_field_magnitudes[i] * Constants::mu_B * mu_s[i];
		}

		// Generate Exchange neighbours
		exchange_neighbours = Neighbours::Get_Neighbours_in_Shells(*geometry, exchange_magnitudes.size());

		// Generate DMI neighbours and normals
		dmi_neighbours = Neighbours::Get_Neighbours_in_Shells(*geometry, dmi_magnitudes.size());
		for (unsigned int ineigh = 0; ineigh < dmi_neighbours.size(); ++ineigh)
		{
			dmi_normals.push_back(Neighbours::DMI_Normal_from_Pair(*geometry, { dmi_neighbours[ineigh].iatom, dmi_neighbours[ineigh].ineigh, {dmi_neighbours[ineigh].translations[0], dmi_neighbours[ineigh].translations[1], dmi_neighbours[ineigh].translations[2]} }, dm_chirality));
		}

		// Generate DDI neighbours, magnitudes and normals
		this->ddi_neighbours = Engine::Neighbours::Get_Neighbours_in_Radius(*this->geometry, ddi_radius);
		scalar magnitude;
		Vector3 normal;
		for (unsigned int i=0; i<ddi_neighbours.size(); ++i)
		{
		    Engine::Neighbours::DDI_from_Pair(*this->geometry, {ddi_neighbours[i].iatom, ddi_neighbours[i].ineigh, {ddi_neighbours[i].translations[0], ddi_neighbours[i].translations[1], ddi_neighbours[i].translations[2]}}, magnitude, normal);
			this->ddi_magnitudes.push_back(magnitude);
			this->ddi_normals.push_back(normal);
		}

		this->Update_Energy_Contributions();
	}

	__inline__ __device__ int neigh_cu_get_pair_j(const int * boundary_conditions, const int * n_cells, int N, int ispin, Neighbour neigh)
	{
		// TODO: use pair.i and pair.j to get multi-spin basis correctly

		// Number of cells
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];

		// Translations (cell) of spin i
		// int ni[3];
		int nic = ispin/(N*Na*Nb);
		int nib = (ispin-nic*N*Na*Nb)/(N*Na);
		int nia = ispin-nic*N*Na*Nb-nib*N*Na;

		// Translations (cell) of spin j (possibly outside of non-periodical domain)
		// int nj[3]
		int nja = nia+neigh.translations[0];
		int njb = nib+neigh.translations[1];
		int njc = nic+neigh.translations[2];

		if ( boundary_conditions[0] || (0 <= nja && nja < Na) )
		{
			// Boundary conditions fulfilled
			// Find the translations of spin j within the non-periodical domain
			if (nja < 0)
				nja += Na;
			// Calculate the correct index
			if (nja>=Na)
				nja-=Na;
		}
		else
		{
			// Boundary conditions not fulfilled
			return -1;
		}

		if ( boundary_conditions[1] || (0 <= njb && njb < Nb) )
		{
			// Boundary conditions fulfilled
			// Find the translations of spin j within the non-periodical domain
			if (njb < 0)
				njb += Nb;
			// Calculate the correct index
			if (njb>=Nb)
				njb-=Nb;
		}
		else
		{
			// Boundary conditions not fulfilled
			return -1;
		}

		if ( boundary_conditions[2] || (0 <= njc && njc < Nc) )
		{
			// Boundary conditions fulfilled
			// Find the translations of spin j within the non-periodical domain
			if (njc < 0)
				njc += Nc;
			// Calculate the correct index
			if (njc>=Nc)
				njc-=Nc;
		}
		else
		{
			// Boundary conditions not fulfilled
			return -1;
		}

		return (nja)*N + (njb)*N*Na + (njc)*N*Na*Nb;
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

	__global__ void HNeigh_CU_E_Zeeman(const Vector3 * spins, const int * atom_types, const int * external_field_indices, const scalar * external_field_magnitudes, const Vector3 * external_field_normals, scalar * Energy, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = external_field_indices[idx];
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (atom_types[ispin] >= 0)
			#endif
			atomicAdd(&Energy[ispin], - external_field_magnitudes[idx] * external_field_normals[idx].dot(spins[ispin]));
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		int size = this->external_field_indices.size();
		HNeigh_CU_E_Zeeman<<<(size+1023)/1024, 1024>>>(spins.data(), this->atom_types.data(), this->external_field_indices.data(), this->external_field_magnitudes.data(), this->external_field_normals.data(), Energy.data(), size);
	}

	__global__ void HNeigh_CU_E_Anisotropy(const Vector3 * spins, const int * atom_types, const int * anisotropy_indices, const scalar * anisotropy_magnitudes, const Vector3 * anisotropy_normals, scalar * Energy, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = anisotropy_indices[idx];
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (atom_types[ispin] >= 0)
			#endif
			atomicAdd(&Energy[ispin], - anisotropy_magnitudes[idx] * std::pow(anisotropy_normals[idx].dot(spins[ispin]), 2.0));
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
	{
		int size = this->anisotropy_indices.size();
		HNeigh_CU_E_Anisotropy<<<(size+1023)/1024, 1024>>>(spins.data(), this->atom_types.data(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), Energy.data(), size);
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
				int jspin = neigh_cu_get_pair_j(bc, nc, n_basis_spins, ispin, neighbours[ineigh]);
				#ifdef SPIRIT_ENABLE_DEFECTS
				if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
				{
				#endif
				int ishell = neighbours[ineigh].idx_shell;
				if ( jspin >= 0 )
				{
					Energy[ispin] -= 0.5 * magnitudes[ishell] * spins[ispin].dot(spins[jspin]);
				}
				#ifdef SPIRIT_ENABLE_DEFECTS
				}
				#endif
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Exchange(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		HNeigh_CU_E_Exchange<<<(size+1023)/1024, 1024>>>( spins.data(), this->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
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
				int jspin = neigh_cu_get_pair_j(bc, nc, n_basis_spins, ispin, neighbours[ineigh]);
				#ifdef SPIRIT_ENABLE_DEFECTS
				if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
				{
				#endif
				int ishell = neighbours[ineigh].idx_shell;
				if ( jspin >= 0 )
				{
					Energy[ispin] -= 0.5 * magnitudes[ishell] * normals[ineigh].dot(spins[ispin].cross(spins[jspin]));
				}
				#ifdef SPIRIT_ENABLE_DEFECTS
				}
				#endif
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_DMI(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		HNeigh_CU_E_DMI<<<(size+1023)/1024, 1024>>>( spins.data(), this->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
				this->dmi_neighbours.size(), this->dmi_neighbours.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), Energy.data(), size );
	}

	void Hamiltonian_Heisenberg_Neighbours::E_DDI(const vectorfield & spins, scalarfield & Energy)
	{
		// //scalar mult = -Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		// scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		// scalar result = 0.0;

		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
		// 	{
		// 		if (ddi_magnitudes[ineigh] > 0.0)
		// 		{
		// 			auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
		// 			if ( Vectormath::boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, ddi_neighbours[ineigh].translations) )
		// 			{
		// 				int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, ddi_neighbours[ineigh].translations);

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
	}

	__global__ void HNeigh_CU_Gradient_Zeeman( const int * atom_types, const int * external_field_indices, const scalar * external_field_magnitude, const Vector3 * external_field_normal, Vector3 * gradient, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = external_field_indices[idx];
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (atom_types[ispin] >= 0)
			{
			#endif
			for (int dim=0; dim<3 ; dim++)
			{
				atomicAdd(&gradient[ispin][dim], -external_field_magnitude[idx]*external_field_normal[idx][dim]);
			}
			#ifdef SPIRIT_ENABLE_DEFECTS
			}
			#endif
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Zeeman(vectorfield & gradient)
	{
		int size = this->external_field_indices.size();
		HNeigh_CU_Gradient_Zeeman<<<(size+1023)/1024, 1024>>>( this->atom_types.data(), this->external_field_indices.data(), this->external_field_magnitudes.data(), this->external_field_normals.data(), gradient.data(), size );
	}

	__global__ void HNeigh_CU_Gradient_Anisotropy(const Vector3 * spins, const int * atom_types, const int * anisotropy_indices, const scalar * anisotropy_magnitudes, const Vector3 * anisotropy_normals, Vector3 * gradient, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = anisotropy_indices[idx];
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (atom_types[ispin] >= 0)
			{
			#endif
			scalar sc = -2 * anisotropy_magnitudes[idx] * anisotropy_normals[idx].dot(spins[ispin]);
			for (int dim=0; dim<3 ; dim++)
			{
				atomicAdd(&gradient[ispin][dim], sc*anisotropy_normals[idx][dim]);
			}
			#ifdef SPIRIT_ENABLE_DEFECTS
			}
			#endif
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
	{
		int size = this->anisotropy_indices.size();
		HNeigh_CU_Gradient_Anisotropy<<<(size+1023)/1024, 1024>>>( spins.data(), this->atom_types.data(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), gradient.data(), size );
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
				int jspin = neigh_cu_get_pair_j(bc, nc, n_basis_spins, ispin, neighbours[ineigh]);
				#ifdef SPIRIT_ENABLE_DEFECTS
				if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
				{
				#endif
				int ishell = neighbours[ineigh].idx_shell;
				if ( jspin >= 0 )
				{
					grad -= magnitudes[ishell] * spins[jspin];
				}
				#ifdef SPIRIT_ENABLE_DEFECTS
				}
				#endif
			}
			gradient[ispin] += grad;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		HNeigh_CU_Gradient_Exchange<<<(size+1023)/1024, 1024>>>( spins.data(), this->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
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
				int jspin = neigh_cu_get_pair_j(bc, nc, n_basis_spins, ispin, neighbours[ineigh]);
				#ifdef SPIRIT_ENABLE_DEFECTS
				if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
				{
				#endif
				int ishell = neighbours[ineigh].idx_shell;
				if ( jspin >= 0 )
				{
					grad -= magnitudes[ishell]*spins[jspin].cross(normals[ineigh]);
				}
				#ifdef SPIRIT_ENABLE_DEFECTS
				}
				#endif
			}
			gradient[ispin] += grad;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		HNeigh_CU_Gradient_DMI<<<(size+1023)/1024, 1024>>>( spins.data(), this->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
				this->dmi_neighbours.size(), this->dmi_neighbours.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), gradient.data(), size );
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
	{
		// //scalar mult = Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		// scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		
		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
		// 	{
		// 		if (ddi_magnitudes[ineigh] > 0.0)
		// 		{
		// 			// std::cerr << ineigh << std::endl;
		// 			auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
		// 			if ( Vectormath::boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, ddi_neighbours[ineigh].translations) )
		// 			{
		// 				int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, ddi_neighbours[ineigh].translations);

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
		// 	auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
		// 	for (unsigned int ineigh = 0; ineigh < this->exchange_neighbours.size(); ++ineigh)
		// 	{
		// 		for (int alpha = 0; alpha < 3; ++alpha)
		// 		{
		// 			//int idx_i = 3 * exchange_neighbours[i_pair][0] + alpha;
		// 			//int idx_j = 3 * exchange_neighbours[i_pair][1] + alpha;
		// 			int jspin = Vectormath::idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, exchange_neighbours[ineigh].translations);
		// 			int ishell = exchange_neighbours[ineigh].idx_shell;
		// 			hessian(ispin, jspin) += -exchange_magnitudes[ineigh];
		// 			hessian(jspin, ispin) += -exchange_magnitudes[ineigh];
		// 		}
		// 	}
		// }
		// // DMI
		// for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		// {
		// 	auto translations = Vectormath::translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
		// 	for (unsigned int ineigh = 0; ineigh < this->dmi_neighbours.size(); ++ineigh)
		// 	{
		// 		for (int alpha = 0; alpha < 3; ++alpha)
		// 		{
		// 			for (int beta = 0; beta < 3; ++beta)
		// 			{
		// 				int idx_i = 3 * dmi_neighbours[ineigh].iatom + alpha;
		// 				int idx_j = 3 * dmi_neighbours[ineigh].ineigh + beta;
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