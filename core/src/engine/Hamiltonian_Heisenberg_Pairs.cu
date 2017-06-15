#ifdef USE_CUDA

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>

#include <engine/Hamiltonian_Heisenberg_Pairs.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>

using std::vector;
using std::function;

using namespace Data;
using namespace Utility;

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
		Pair pair;
		scalar magnitude;
		Vector3 normal;
		for (unsigned int i = 0; i<ddi_pairs.size(); ++i)
		{
			pair = Pair{ ddi_pairs[i].i, ddi_pairs[i].j, {ddi_pairs[i].translations[0], ddi_pairs[i].translations[1], ddi_pairs[i].translations[2]} };
			Engine::Neighbours::DDI_from_Pair(*this->geometry, pair, magnitude, normal);
			this->ddi_magnitudes.push_back(magnitude);
			this->ddi_normals.push_back(normal);
		}

		this->Update_Energy_Contributions();
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
		// Quadruplet
		if (this->quadruplets.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"Quadruplet", scalarfield(0) });
			this->idx_quadruplet = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_quadruplet = -1;
	}

	inline int idx_from_translations(const intfield & n_cells, const int n_spins_basic_domain, const std::array<int, 3> & translations)
	{
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];
		int N = n_spins_basic_domain;

		int da = translations[0];
		int db = translations[1];
		int dc = translations[2];

		return da*N + db*N*Na + dc*N*Na*Nb;
	}

	inline int idx_from_translations(const intfield & n_cells, const int n_spins_basic_domain, const std::array<int, 3> & translations_i, const int translations[3])
	{
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];
		int N = n_spins_basic_domain;

		int da = translations_i[0] + translations[0];
		int db = translations_i[1] + translations[1];
		int dc = translations_i[2] + translations[2];

		if (translations[0] < 0)
			da += N*Na;
		if (translations[1] < 0)
			db += N*Na*Nb;
		if (translations[2] < 0)
			dc += N*Na*Nb*Nc;

		int idx = (da%Na)*N + (db%Nb)*N*Na + (dc%Nc)*N*Na*Nb;

		return idx;
	}

	__inline__ __device__ int pair_cu_get_pair_j(const int * boundary_conditions, const int * n_cells, int N, int ispin, Pair pair)
	{
		// TODO: use pair.i and pair.j to get multi-spin basis correctly

		// Number of cells
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];

		// Translations (cell index) of spin i
		// int ni[3];
		int nic = ispin/(N*Na*Nb);
		int nib = (ispin-nic*N*Na*Nb)/(N*Na);
		int nia = ispin-nic*N*Na*Nb-nib*N*Na;

		// Translations (cell index) of spin j (possibly outside of non-periodical domain)
		// int nj[3]
		int nja = nia+pair.translations[0];
		int njb = nib+pair.translations[1];
		int njc = nic+pair.translations[2];

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


	void Hamiltonian_Heisenberg_Pairs::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
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
		
		cudaDeviceSynchronize();
	}

	
	__global__ void CU_E_Zeeman(const Vector3 * spins, const int * atom_types, const int * external_field_indices, const scalar * external_field_magnitude, const Vector3 * external_field_normal, scalar * Energy, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = external_field_indices[idx];
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (atom_types[ispin] >= 0)
			#endif
			atomicAdd(&Energy[ispin], - external_field_magnitude[idx] * external_field_normal[idx].dot(spins[ispin]));
		}
	}
	void Hamiltonian_Heisenberg_Pairs::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		int size = this->external_field_indices.size();
		CU_E_Zeeman<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), this->external_field_indices.data(), this->external_field_magnitudes.data(), this->external_field_normals.data(), Energy.data(), size);
	}


	__global__ void CU_E_Anisotropy(const Vector3 * spins, const int * atom_types, const int * anisotropy_indices, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, scalar * Energy, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = anisotropy_indices[idx];
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (atom_types[ispin] >= 0)
			#endif
			atomicAdd(&Energy[ispin], - anisotropy_magnitude[idx] * std::pow(anisotropy_normal[idx].dot(spins[ispin]), 2.0));
		}
	}
	void Hamiltonian_Heisenberg_Pairs::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
	{
		int size = this->anisotropy_indices.size();
		CU_E_Anisotropy<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), Energy.data(), size);
	}


	__global__ void CU_E_Exchange(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_pairs, const Pair * pairs, const scalar * magnitudes, scalar * Energy, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			for(auto ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = pair_cu_get_pair_j(bc, nc, n_basis_spins, ispin, pairs[ipair]);
				if (jspin >= 0)
				{
					#ifdef SPIRIT_ENABLE_DEFECTS
					if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
					{
					#endif
					scalar sc = - 0.5 * magnitudes[ipair] * spins[ispin].dot(spins[jspin]);
					atomicAdd(&Energy[ispin], sc);
					atomicAdd(&Energy[jspin], sc);
					#ifdef SPIRIT_ENABLE_DEFECTS
					}
					#endif
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Pairs::E_Exchange(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		CU_E_Exchange<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
				this->exchange_pairs.size(), this->exchange_pairs.data(), this->exchange_magnitudes.data(), Energy.data(), size);
	}


	__global__ void CU_E_DMI(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals, scalar * Energy, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};
		
		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			for(auto ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = pair_cu_get_pair_j(bc, nc, n_basis_spins, ispin, pairs[ipair]);
				if (jspin >= 0)
				{
					#ifdef SPIRIT_ENABLE_DEFECTS
					if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
					{
					#endif
					scalar sc = - 0.5 * magnitudes[ipair] * normals[ipair].dot(spins[ispin].cross(spins[jspin]));
					atomicAdd(&Energy[ispin], sc);
					atomicAdd(&Energy[jspin], sc);
					#ifdef SPIRIT_ENABLE_DEFECTS
					}
					#endif
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Pairs::E_DMI(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		CU_E_DMI<<<(size+1023)/1024, 1024>>>(spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
				this->dmi_pairs.size(), this->dmi_pairs.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), Energy.data(), size);
	}


	void Hamiltonian_Heisenberg_Pairs::E_DDI(const vectorfield & spins, scalarfield & Energy)
	{
		// //scalar mult = -Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
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
		// 					int idx_i = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
		// 					int idx_j = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, ddi_pairs[i_pair].translations);
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


	void Hamiltonian_Heisenberg_Pairs::E_Quadruplet(const vectorfield & spins, scalarfield & Energy)
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
		// 				int i = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
		// 				int j = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_j);
		// 				int k = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_k);
		// 				int l = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_l);
		// 				Energy[i] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
		// 				Energy[j] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
		// 				Energy[k] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
		// 				Energy[l] -= 0.25*quadruplet_magnitudes[iquad] * (spins[i].dot(spins[j])) * (spins[k].dot(spins[l]));
		// 			}
		// 		}
		// 	}
		// }
	}



	void Hamiltonian_Heisenberg_Pairs::Gradient(const vectorfield & spins, vectorfield & gradient)
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

		cudaDeviceSynchronize();
	}


	__global__ void CU_Gradient_Zeeman( const int * atom_types, const int * external_field_indices, const scalar * external_field_magnitude, const Vector3 * external_field_normal, Vector3 * gradient, size_t size)
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
	void Hamiltonian_Heisenberg_Pairs::Gradient_Zeeman(vectorfield & gradient)
	{
		int size = this->external_field_indices.size();
		CU_Gradient_Zeeman<<<(size+1023)/1024, 1024>>>( this->geometry->atom_types.data(), this->external_field_indices.data(), this->external_field_magnitudes.data(), this->external_field_normals.data(), gradient.data(), size );
	}


	__global__ void CU_Gradient_Anisotropy(const Vector3 * spins, const int * atom_types, const int * anisotropy_indices, const scalar * anisotropy_magnitudes, const Vector3 * anisotropy_normals, Vector3 * gradient, size_t size)
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
	void Hamiltonian_Heisenberg_Pairs::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
	{
		int size = this->anisotropy_indices.size();
		CU_Gradient_Anisotropy<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), this->anisotropy_indices.data(), this->anisotropy_magnitudes.data(), this->anisotropy_normals.data(), gradient.data(), size );
	}


	__global__ void CU_Gradient_Exchange(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_pairs, const Pair * pairs, const scalar * magnitudes, Vector3 * gradient, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};

		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			for(auto ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = pair_cu_get_pair_j(bc, nc, n_basis_spins, ispin, pairs[ipair]);
				if (jspin >= 0)
				{
					#ifdef SPIRIT_ENABLE_DEFECTS
					if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
					{
					#endif
					for (int dim=0; dim<3 ; dim++)
					{
						atomicAdd(&gradient[ispin][dim], -magnitudes[ipair]*spins[jspin][dim]);
						atomicAdd(&gradient[jspin][dim], -magnitudes[ipair]*spins[ispin][dim]);
					}
					#ifdef SPIRIT_ENABLE_DEFECTS
					}
					#endif
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Pairs::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		CU_Gradient_Exchange<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
				this->exchange_pairs.size(), this->exchange_pairs.data(), this->exchange_magnitudes.data(), gradient.data(), size );
	}


	__global__ void CU_Gradient_DMI(const Vector3 * spins, const int * atom_types, const int * boundary_conditions, const int * n_cells, int n_basis_spins,
			int n_pairs, const Pair * pairs, const scalar * magnitudes, const Vector3 * normals, Vector3 * gradient, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};
		
		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			for(auto ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = pair_cu_get_pair_j(bc, nc, n_basis_spins, ispin, pairs[ipair]);
				if (jspin >= 0)
				{
					#ifdef SPIRIT_ENABLE_DEFECTS
					if (atom_types[ispin] >= 0 && atom_types[jspin] >= 0)
					{
					#endif
					Vector3 jcross = magnitudes[ipair]*spins[jspin].cross(normals[ipair]);
					Vector3 icross = magnitudes[ipair]*spins[ispin].cross(normals[ipair]);
					for (int dim=0; dim<3 ; dim++)
					{
						atomicAdd(&gradient[ispin][dim], -jcross[dim]);
						atomicAdd(&gradient[jspin][dim],  icross[dim]);
					}
					#ifdef SPIRIT_ENABLE_DEFECTS
					}
					#endif
				}
			}
		}
	}
	void Hamiltonian_Heisenberg_Pairs::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		CU_Gradient_DMI<<<(size+1023)/1024, 1024>>>( spins.data(), this->geometry->atom_types.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain,
				this->dmi_pairs.size(),  this->dmi_pairs.data(), this->dmi_magnitudes.data(), this->dmi_normals.data(), gradient.data(), size );
	}


	void Hamiltonian_Heisenberg_Pairs::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
	{
		// //scalar mult = Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
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
		// 						int ispin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
		// 						int jspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, ddi_pairs[i_pair].translations);
		// 						gradient[ispin] -= skalar_contrib * (3 * ddi_normals[i_pair] * spins[jspin].dot(ddi_normals[i_pair]) - spins[jspin]);
		// 						gradient[jspin] -= skalar_contrib * (3 * ddi_normals[i_pair] * spins[ispin].dot(ddi_normals[i_pair]) - spins[ispin]);
		// 					}
		// 				}
		// 			}
		// 		}
		// 	}
		// }
	}//end Field_DipoleDipole


	void Hamiltonian_Heisenberg_Pairs::Gradient_Quadruplet(const vectorfield & spins, vectorfield & gradient)
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
		// 				int ispin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations);
		// 				int jspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_j);
		// 				int kspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_k);
		// 				int lspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, quadruplets[iquad].d_l);
		// 				gradient[ispin] -= quadruplet_magnitudes[iquad] * spins[jspin] * (spins[kspin].dot(spins[lspin]));
		// 				gradient[jspin] -= quadruplet_magnitudes[iquad] * spins[ispin] * (spins[kspin].dot(spins[lspin]));
		// 				gradient[kspin] -= quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * spins[lspin];
		// 				gradient[lspin] -= quadruplet_magnitudes[iquad] * (spins[ispin].dot(spins[jspin])) * spins[kspin];
		// 			}
		// 		}
		// 	}
		// }
	}


	void Hamiltonian_Heisenberg_Pairs::Hessian(const vectorfield & spins, MatrixX & hessian)
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
							int idx_i = 3 * idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations) + alpha;
							int idx_j = 3 * idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, exchange_pairs[i_pair].translations) + alpha;
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
								int idx_i = 3 * idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations) + alpha;
								int idx_j = 3 * idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, dmi_pairs[i_pair].translations) + alpha;
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
	static const std::string name = "Heisenberg (Pairs)";
	const std::string& Hamiltonian_Heisenberg_Pairs::Name() { return name; }
}

#endif