#ifdef USE_CUDA

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>

#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>

using std::vector;
using std::function;

using namespace Data;
using namespace Utility;

namespace Engine
{
	Hamiltonian_Heisenberg_Neighbours::Hamiltonian_Heisenberg_Neighbours(
			scalarfield mu_s,
			intfield external_field_index, scalarfield external_field_magnitude, vectorfield external_field_normal,
			intfield anisotropy_index, scalarfield anisotropy_magnitude, vectorfield anisotropy_normal,
			pairfield Exchange_pairs, scalarfield Exchange_magnitude,
			pairfield DMI_pairs, scalarfield DMI_magnitude, vectorfield DMI_normal,
			pairfield DD_pairs, scalarfield DD_magnitude, vectorfield DD_normal,
			quadrupletfield quadruplets, scalarfield quadruplet_magnitude,
			std::shared_ptr<Data::Geometry> geometry,
			intfield boundary_conditions
	) :
		Hamiltonian(boundary_conditions),
		geometry(geometry),
		mu_s(mu_s),
		external_field_index(external_field_index), external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
		anisotropy_index(anisotropy_index), anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
		Exchange_pairs(Exchange_pairs), Exchange_magnitude(Exchange_magnitude),
		DMI_pairs(DMI_pairs), DMI_magnitude(DMI_magnitude), DMI_normal(DMI_normal),
		DD_pairs(DD_pairs), DD_magnitude(DD_magnitude), DD_normal(DD_normal),
		quadruplets(quadruplets), quadruplet_magnitude(quadruplet_magnitude)
	{
		// Renormalize the external field from Tesla to whatever
		for (unsigned int i = 0; i < external_field_magnitude.size(); ++i)
		{
			this->external_field_magnitude[i] = this->external_field_magnitude[i] * Constants::mu_B * mu_s[i];
		}

		this->Update_Energy_Contributions();
	}

	void Hamiltonian_Heisenberg_Neighbours::Update_Energy_Contributions()
	{
		this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

		// External field
		if (this->external_field_index.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
			this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_zeeman = -1;
		// Anisotropy
		if (this->anisotropy_index.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0) });
			this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_anisotropy = -1;
		// Exchange
		if (this->Exchange_pairs.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
			this->idx_exchange = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_exchange = -1;
		// DMI
		if (this->DMI_pairs.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
			this->idx_dmi = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_dmi = -1;
		// Dipole-Dipole
		if (this->DD_pairs.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"DD", scalarfield(0) });
			this->idx_dd = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_dd = -1;
		// Quadruplet
		if (this->quadruplets.size() > 0)
		{
			this->energy_contributions_per_spin.push_back({"Quadruplet", scalarfield(0) });
			this->idx_quadruplet = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_quadruplet = -1;
	}


	void Hamiltonian_Heisenberg_Neighbours::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
	{
		int nos = spins.size();
		for (auto& pair : energy_contributions_per_spin)
		{
			// Allocate if not already allocated
			if (pair.second.size() != nos) pair.second = scalarfield(nos, 0);
			// Otherwise set to zero
			else for (auto& pair : energy_contributions_per_spin) Vectormath::fill(pair.second, 0);
		}
		

		// External field
		if (this->idx_zeeman >=0 ) E_Zeeman(spins, energy_contributions_per_spin[idx_zeeman].second);

		// Anisotropy
		if (this->idx_anisotropy >=0 ) E_Anisotropy(spins, energy_contributions_per_spin[idx_anisotropy].second);

		// Pairs
		// Exchange
		if (this->idx_exchange >=0 )   E_Exchange(spins,energy_contributions_per_spin[idx_exchange].second);
		// DMI
		if (this->idx_dmi >=0 )        E_DMI(spins, energy_contributions_per_spin[idx_dmi].second);
		// DD
		if (this->idx_dd >=0 )         E_DD(spins, energy_contributions_per_spin[idx_dd].second);
		// Quadruplet
		if (this->idx_quadruplet >=0 ) E_Quadruplet(spins, energy_contributions_per_spin[idx_quadruplet].second);

		// Return
		//return this->E;
		cudaDeviceSynchronize();
	}


	__inline__ __device__ int cu_get_pair_j(const int * boundary_conditions, const int * n_cells, int N, int ispin, Pair pair)
	{
		// TODO: use pair.i and pair.j to get multi-spin basis correctly

		// Number of cells
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];

		// Translations of spin i
		// int ni[3];
		int nic = ispin/(Na*Nb);
		int nib = (ispin-nic*Na*Nb)/Na;
		int nia = ispin-nic*Na*Nb-nib*Na;

		// Translations of spin j (possibly outside of non-periodical domain)
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

		if ( ( boundary_conditions[0] || (0 <= nja && nja < Na) ) &&
		     ( boundary_conditions[1] || (0 <= njb && njb < Nb) ) &&
		     ( boundary_conditions[2] || (0 <= njc && njc < Nc) ) )
		{
			if (njb < 0)
				njb += Nb;
			if (njc < 0)
				njc += Nc;
			if (njb>=Nb)
				njb-=Nb;
			if (njc>=Nc)
				njc-=Nc;
		}

		if ( ( boundary_conditions[0] || (0 <= nja && nja < Na) ) &&
		     ( boundary_conditions[1] || (0 <= njb && njb < Nb) ) &&
		     ( boundary_conditions[2] || (0 <= njc && njc < Nc) ) )
		{
		}

		return (nja)*N + (njb)*N*Na + (njc)*N*Na*Nb;
	}
	
	__global__ void CU_E_Zeeman(const Vector3 * spins, const int * external_field_index, const scalar * external_field_magnitude, const Vector3 * external_field_normal, scalar * Energy, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			atomicAdd(&Energy[external_field_index[idx]], - external_field_magnitude[idx] * external_field_normal[idx].dot(spins[external_field_index[idx]]));
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		int size = this->external_field_index.size();
		CU_E_Zeeman<<<(size+1023)/1024, 1024>>>(spins.data(), this->external_field_index.data(), this->external_field_magnitude.data(), this->external_field_normal.data(), Energy.data(), size);
	}


	__global__ void CU_E_Anisotropy(const Vector3 * spins, const int * anisotropy_index, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, scalar * Energy, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			atomicAdd(&Energy[anisotropy_index[idx]], - anisotropy_magnitude[idx] * std::pow(anisotropy_normal[idx].dot(spins[anisotropy_index[idx]]), 2.0));
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
	{
		int size = this->anisotropy_index.size();
		CU_E_Anisotropy<<<(size+1023)/1024, 1024>>>(spins.data(), this->anisotropy_index.data(), this->anisotropy_magnitude.data(), this->anisotropy_normal.data(), Energy.data(), size);
	}


	__global__ void CU_E_Exchange(const Vector3 * spins, const int * boundary_conditions, const int * n_cells, int n_basis_spins, const scalar * Exchange_magnitude, const Pair * Exchange_pairs, int n_pairs, scalar * Energy, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};
		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			float E=0;
			Vector3 spin=spins[ispin]; 
			for (unsigned int ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = cu_get_pair_j(bc, nc, n_basis_spins, ispin, Exchange_pairs[ipair]);
				if ( jspin >= 0 )
				{
					E -= 0.5 * Exchange_magnitude[ipair] * spin.dot(spins[jspin]);
				}
			}
			Energy[ispin]+=E;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_Exchange(const vectorfield & spins, scalarfield & Energy)
	{
		int nos = spins.size();
		CU_E_Exchange<<<(nos+1023)/1024, 1024>>>(spins.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain[0], Exchange_magnitude.data(), Exchange_pairs.data(), Exchange_pairs.size(), Energy.data(), nos);
	}


	__global__ void CU_E_DMI(const Vector3 * spins, const int * boundary_conditions, const int * n_cells, int n_basis_spins, const scalar * DMI_magnitude, const Vector3 * DMI_normal, const Pair * DMI_pairs, int n_pairs, scalar * Energy, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};
		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			float E=0;
			Vector3 spin=spins[ispin]; 
			for (unsigned int ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = cu_get_pair_j(bc, nc, n_basis_spins, ispin, DMI_pairs[ipair]);
				if ( jspin >= 0 )
				{
					E -= 0.5;// *  DMI_magnitude[ipair] * DMI_normal[ipair].dot(spin.cross(spins[jspin]));
				}
			}
			Energy[ispin] += E;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::E_DMI(const vectorfield & spins, scalarfield & Energy)
	{
		int size = spins.size();
		CU_E_DMI<<<(size+255)/256, 256>>>(spins.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain[0], DMI_magnitude.data(), DMI_normal.data(), DMI_pairs.data(), DMI_pairs.size(), Energy.data(), size);
	}


	void Hamiltonian_Heisenberg_Neighbours::E_DD(const vectorfield & spins, scalarfield & Energy)
	{
		//scalar mult = -Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		// scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		// scalar result = 0.0;

		for (unsigned int i_pair = 0; i_pair < DD_pairs.size(); ++i_pair)
		{
			if (DD_magnitude[i_pair] > 0.0)
			{
				// Energy[pairs[i_pair][0]] -= mult / std::pow(DD_magnitude[i_pair], 3.0) *
				// 	(3 * spins[pairs[i_pair][1]].dot(DD_normal[i_pair]) * spins[pairs[i_pair][0]].dot(DD_normal[i_pair]) - spins[pairs[i_pair][0]].dot(spins[pairs[i_pair][1]]));
				// Energy[pairs[i_pair][1]] -= mult / std::pow(DD_magnitude[i_pair], 3.0) *
				// 	(3 * spins[pairs[i_pair][1]].dot(DD_normal[i_pair]) * spins[pairs[i_pair][0]].dot(DD_normal[i_pair]) - spins[pairs[i_pair][0]].dot(spins[pairs[i_pair][1]]));
			}

		}
	}// end DipoleDipole


	void Hamiltonian_Heisenberg_Neighbours::E_Quadruplet(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int i_pair = 0; i_pair < quadruplets.size(); ++i_pair)
		{
			// Energy[pairs[i_pair][0]] -= 0.25*magnitude[i_pair] * (spins[pairs[i_pair][0]].dot(spins[pairs[i_pair][1]])) * (spins[pairs[i_pair][2]].dot(spins[pairs[i_pair][3]]));
			// Energy[pairs[i_pair][1]] -= 0.25*magnitude[i_pair] * (spins[pairs[i_pair][0]].dot(spins[pairs[i_pair][1]])) * (spins[pairs[i_pair][2]].dot(spins[pairs[i_pair][3]]));
			// Energy[pairs[i_pair][2]] -= 0.25*magnitude[i_pair] * (spins[pairs[i_pair][0]].dot(spins[pairs[i_pair][1]])) * (spins[pairs[i_pair][2]].dot(spins[pairs[i_pair][3]]));
			// Energy[pairs[i_pair][3]] -= 0.25*magnitude[i_pair] * (spins[pairs[i_pair][0]].dot(spins[pairs[i_pair][1]])) * (spins[pairs[i_pair][2]].dot(spins[pairs[i_pair][3]]));
		}
	}



	void Hamiltonian_Heisenberg_Neighbours::Gradient(const vectorfield & spins, vectorfield & gradient)
	{
		// Set to zero
		Vectormath::fill(gradient, {0,0,0});

		// External field
		Gradient_Zeeman(gradient);

		// Anisotropy
		Gradient_Anisotropy(spins, gradient);

		// Pairs
		// Exchange
		this->Gradient_Exchange(spins, gradient);
		// DMI
		this->Gradient_DMI(spins, gradient);
		// DD
		this->Gradient_DD(spins, gradient);

		// Triplet

		// Quadruplet
		this->Gradient_Quadruplet(spins, gradient);

		cudaDeviceSynchronize();
	}


	__global__ void CU_Gradient_Zeeman( const int * external_field_index, const scalar * external_field_magnitude, const Vector3 * external_field_normal, Vector3 * gradient, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = external_field_index[idx];
			for (int dim=0; dim<3 ; dim++)
			{
				atomicAdd(&gradient[ispin][dim], -external_field_magnitude[idx]*external_field_normal[idx][dim]);
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Zeeman(vectorfield & gradient)
	{
		int size = this->external_field_index.size();
		CU_Gradient_Zeeman<<<(size+1023)/1024, 1024>>>( this->external_field_index.data(), this->external_field_magnitude.data(), this->external_field_normal.data(), gradient.data(), size );
	}


	__global__ void CU_Gradient_Anisotropy(const Vector3 * spins, const int * anisotropy_index, const scalar * anisotropy_magnitude, const Vector3 * anisotropy_normal, Vector3 * gradient, size_t size)
	{
		for(auto idx = blockIdx.x * blockDim.x + threadIdx.x;
			idx < size;
			idx +=  blockDim.x * gridDim.x)
		{
			int ispin = anisotropy_index[idx];
			scalar sc = -2 * anisotropy_magnitude[idx] * anisotropy_normal[idx].dot(spins[ispin]);
			for (int dim=0; dim<3 ; dim++)
			{
				atomicAdd(&gradient[ispin][dim], sc*anisotropy_normal[idx][dim]);
			}
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
	{
		int size = this->anisotropy_index.size();
		CU_Gradient_Anisotropy<<<(size+1023)/1024, 1024>>>( spins.data(), this->anisotropy_index.data(), this->anisotropy_magnitude.data(), this->anisotropy_normal.data(), gradient.data(), size );
	}


	__global__ void CU_Gradient_Exchange(const Vector3 * spins, const int * boundary_conditions, const int * n_cells, int n_basis_spins, const scalar * Exchange_magnitude, const Pair * Exchange_pairs, int n_pairs, Vector3 * gradient, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};
		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			Vector3 grad{0,0,0};
			Vector3 spin=spins[ispin]; 
			for (unsigned int ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = cu_get_pair_j(bc, nc, n_basis_spins, ispin, Exchange_pairs[ipair]);
				if ( jspin >= 0 )
				{
					grad -= Exchange_magnitude[ipair] * spins[jspin];
				}
			}
			gradient[ispin] += grad;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		CU_Gradient_Exchange<<<(size+1023)/1024, 1024>>>(spins.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain[0], Exchange_magnitude.data(), Exchange_pairs.data(), Exchange_pairs.size(), gradient.data(), size);
	}


	__global__ void CU_Gradient_DMI(const Vector3 * spins, const int * boundary_conditions, const int * n_cells, int n_basis_spins, const scalar * DMI_magnitude, const Vector3 * DMI_normal, const Pair * DMI_pairs, int n_pairs, Vector3 * gradient, size_t size)
	{
		int bc[3]={boundary_conditions[0],boundary_conditions[1],boundary_conditions[2]};
		int nc[3]={n_cells[0],n_cells[1],n_cells[2]};
		for(auto ispin = blockIdx.x * blockDim.x + threadIdx.x;
			ispin < size;
			ispin +=  blockDim.x * gridDim.x)
		{
			Vector3 grad{0,0,0};
			Vector3 spin=spins[ispin]; 
			for (unsigned int ipair = 0; ipair < n_pairs; ++ipair)
			{
				int jspin = cu_get_pair_j(bc, nc, n_basis_spins, ispin, DMI_pairs[ipair]);
				if ( jspin >= 0 )
				{
					grad -= DMI_magnitude[ipair]*spins[jspin].cross(DMI_normal[ipair]);
				}
			}
			gradient[ispin] += grad;
		}
	}
	void Hamiltonian_Heisenberg_Neighbours::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
	{
		int size = spins.size();
		CU_Gradient_DMI<<<(size+1023)/1024, 1024>>>(spins.data(), boundary_conditions.data(), geometry->n_cells.data(), geometry->n_spins_basic_domain[0], DMI_magnitude.data(), DMI_normal.data(), DMI_pairs.data(), DMI_pairs.size(), gradient.data(), size);
	}


	void Hamiltonian_Heisenberg_Neighbours::Gradient_DD(const vectorfield & spins, vectorfield & gradient)
	{
		//scalar mult = Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		
		for (unsigned int i_pair = 0; i_pair < DD_pairs.size(); ++i_pair)
		{
			if (DD_magnitude[i_pair] > 0.0)
			{
				scalar skalar_contrib = mult / std::pow(DD_magnitude[i_pair], 3.0);
				// gradient[indices[i_pair][0]] -= skalar_contrib * (3 * DD_normal[i_pair] * spins[indices[i_pair][1]].dot(DD_normal[i_pair]) - spins[indices[i_pair][1]]);
				// gradient[indices[i_pair][1]] -= skalar_contrib * (3 * DD_normal[i_pair] * spins[indices[i_pair][0]].dot(DD_normal[i_pair]) - spins[indices[i_pair][0]]);
			}
		}
	}//end Field_DipoleDipole


	void Hamiltonian_Heisenberg_Neighbours::Gradient_Quadruplet(const vectorfield & spins, vectorfield & gradient)
	{
		for (unsigned int i_pair = 0; i_pair < quadruplets.size(); ++i_pair)
		{
			// gradient[indices[i_pair][0]] -= magnitude[i_pair] * spins[indices[i_pair][1]] * (spins[indices[i_pair][2]].dot(spins[indices[i_pair][3]]));
			// gradient[indices[i_pair][1]] -= magnitude[i_pair] * spins[indices[i_pair][0]] *  (spins[indices[i_pair][2]].dot(spins[indices[i_pair][3]]));
			// gradient[indices[i_pair][2]] -= magnitude[i_pair] * (spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]])) * spins[indices[i_pair][3]];
			// gradient[indices[i_pair][3]] -= magnitude[i_pair] * (spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]])) * spins[indices[i_pair][2]];
		}
	}


	void Hamiltonian_Heisenberg_Neighbours::Hessian(const vectorfield & spins, MatrixX & hessian)
	{
		int nos = spins.size();

		// Set to zero
		// for (auto& h : hessian) h = 0;
		hessian.setZero();

		// Single Spin elements
		for (int alpha = 0; alpha < 3; ++alpha)
		{
			for (unsigned int i = 0; i < anisotropy_index.size(); ++i)
			{
				int idx = anisotropy_index[i];
				// scalar x = -2.0*this->anisotropy_magnitude[i] * std::pow(this->anisotropy_normal[i][alpha], 2);
				hessian(3*idx + alpha, 3*idx + alpha) += -2.0*this->anisotropy_magnitude[i]*std::pow(this->anisotropy_normal[i][alpha],2);
			}
		}

		// std::cerr << "calculated hessian" << std::endl;

		//  // Spin Pair elements
		//  for (int i_periodicity = 0; i_periodicity < 8; ++i_periodicity)
		//  {
		//  	//		Check if boundary conditions contain this periodicity
		//  	if ((i_periodicity == 0)
		//  		|| (i_periodicity == 1 && this->boundary_conditions[0])
		//  		|| (i_periodicity == 2 && this->boundary_conditions[1])
		//  		|| (i_periodicity == 3 && this->boundary_conditions[2])
		//  		|| (i_periodicity == 4 && this->boundary_conditions[0] && this->boundary_conditions[1])
		//  		|| (i_periodicity == 5 && this->boundary_conditions[0] && this->boundary_conditions[2])
		//  		|| (i_periodicity == 6 && this->boundary_conditions[1] && this->boundary_conditions[2])
		//  		|| (i_periodicity == 7 && this->boundary_conditions[0] && this->boundary_conditions[1] && this->boundary_conditions[2]))
		//  	{
		//  		//		Loop over pairs of this periodicity
		//  		// Exchange
		//  		for (unsigned int i_pair = 0; i_pair < this->Exchange_pairs.size(); ++i_pair)
		//  		{
		//  			for (int alpha = 0; alpha < 3; ++alpha)
		//  			{
		//  				int idx_i = 3*Exchange_pairs[i_pair][0] + alpha;
		//  				int idx_j = 3*Exchange_pairs[i_pair][1] + alpha;
		//  				hessian(idx_i,idx_j) += -Exchange_magnitude[i_pair];
		//  				hessian(idx_j,idx_i) += -Exchange_magnitude[i_pair];
		//  			}
		//  		}
		//  		// DMI
		//  		for (unsigned int i_pair = 0; i_pair < this->DMI_pairs[i_periodicity].size(); ++i_pair)
		//  		{
		//  			for (int alpha = 0; alpha < 3; ++alpha)
		//  			{
		//  				for (int beta = 0; beta < 3; ++beta)
		//  				{
		//  					int idx_i = 3*DMI_pairs[i_periodicity][i_pair][0] + alpha;
		//  					int idx_j = 3*DMI_pairs[i_periodicity][i_pair][1] + beta;
		//  					if ( (alpha == 0 && beta == 1) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][2];
		//  						hessian(idx_j,idx_i) +=
		//  							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][2];
		//  					}
		//  					else if ( (alpha == 1 && beta == 0) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][2];
		//  						hessian(idx_j,idx_i) +=
		//  							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][2];
		//  					}
		//  					else if ( (alpha == 0 && beta == 2) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][1];
		//  						hessian(idx_j,idx_i) +=
		//  							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][1];
		//  					}
		//  					else if ( (alpha == 2 && beta == 0) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][1];
		//  						hessian(idx_j,idx_i) +=
		//  							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][1];
		//  					}
		//  					else if ( (alpha == 1 && beta == 2) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][0];
		//  						hessian(idx_j,idx_i) +=
		//  							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][0];
		//  					}
		//  					else if ( (alpha == 2 && beta == 1) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][0];
		//  						hessian(idx_j,idx_i) +=
		//  							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][0];
		//  					}
		//  				}
		//  			}
		//  		}
		//  //		// Dipole-Dipole
		//  //		for (unsigned int i_pair = 0; i_pair < this->DD_pairs[i_periodicity].size(); ++i_pair)
		//  //		{
		//  //			// indices
		//  //			int idx_1 = DD_pairs[i_periodicity][i_pair][0];
		//  //			int idx_2 = DD_pairs[i_periodicity][i_pair][1];
		//  //			// prefactor
		//  //			scalar prefactor = 0.0536814951168
		//  //				* this->mu_s[idx_1] * this->mu_s[idx_2]
		//  //				/ std::pow(DD_magnitude[i_periodicity][i_pair], 3);
		//  //			// components
		//  //			for (int alpha = 0; alpha < 3; ++alpha)
		//  //			{
		//  //				for (int beta = 0; beta < 3; ++beta)
		//  //				{
		//  //					int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
		//  //					if (alpha == beta)
		//  //						hessian[idx_h] += prefactor;
		//  //					hessian[idx_h] += -3.0*prefactor*DD_normal[i_periodicity][i_pair][alpha] * DD_normal[i_periodicity][i_pair][beta];
		//  //				}
		//  //			}
		//  //		}
		//  	}// end if periodicity
		//  }// end for periodicity
	}

	// Hamiltonian name as string
	static const std::string name = "Heisenberg (Neighbours)";
	const std::string& Hamiltonian_Heisenberg_Neighbours::Name() { return name; }
}

#endif