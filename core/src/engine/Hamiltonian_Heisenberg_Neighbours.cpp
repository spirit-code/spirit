#ifndef USE_CUDA

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>

#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
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
	inline bool boundary_conditions_fulfilled(const intfield & n_cells, const intfield & boundary_conditions, const std::array<int,3> & translations_i, const std::array<int,3> & translations_j)
	{
		int da = translations_i[0]+translations_j[0];
		int db = translations_i[1]+translations_j[1];
		int dc = translations_i[2]+translations_j[2];
		return  ( ( boundary_conditions[0] || (0 <= da && da < n_cells[0]) ) &&
				  ( boundary_conditions[1] || (0 <= db && db < n_cells[1]) ) &&
				  ( boundary_conditions[2] || (0 <= dc && dc < n_cells[2]) ) );
	}

	inline int idx_from_translations(const intfield & n_cells, const int n_spins_basic_domain, const std::array<int,3> & translations_i, const std::array<int,3> & translations)
	{
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];
		int N  = n_spins_basic_domain;
		
		int da = translations_i[0]+translations[0];
		int db = translations_i[1]+translations[1];
		int dc = translations_i[2]+translations[2];

		if (translations[0] < 0)
			da += N*Na;
		if (translations[1] < 0)
			db += N*Na*Nb;
		if (translations[2] < 0)
			dc += N*Na*Nb*Nc;
			
		int idx = (da%Na)*N + (db%Nb)*N*Na + (dc%Nc)*N*Na*Nb;
		
		return idx;
	}

	inline std::array<int,3> translations_from_idx(const intfield & n_cells, const int n_spins_basic_domain, int idx)
	{
		std::array<int,3> ret;
		int Na = n_cells[0];
		int Nb = n_cells[1];
		int Nc = n_cells[2];
		int N  = n_spins_basic_domain;
		ret[2] = idx/(Na*Nb);
		ret[1] = (idx-ret[2]*Na*Nb)/Na;
		ret[0] = idx-ret[2]*Na*Nb-ret[1]*Na;
		return ret;
	}

	Hamiltonian_Heisenberg_Neighbours::Hamiltonian_Heisenberg_Neighbours(
			scalarfield mu_s,
			intfield external_field_index, scalarfield external_field_magnitude, vectorfield external_field_normal,
			intfield anisotropy_index, scalarfield anisotropy_magnitude, vectorfield anisotropy_normal,
			scalarfield exchange_magnitude,
			scalarfield dmi_magnitude, int dm_chirality,
			scalar ddi_radius,
			std::shared_ptr<Data::Geometry> geometry,
			intfield boundary_conditions
	) :
		Hamiltonian(boundary_conditions),
		geometry(geometry),
		mu_s(mu_s),
		external_field_index(external_field_index), external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
		anisotropy_index(anisotropy_index), anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
		exchange_magnitude(exchange_magnitude),
		dmi_magnitude(dmi_magnitude),
		ddi_radius(ddi_radius)
	{
		// Renormalize the external field from Tesla to meV
		for (unsigned int i = 0; i < external_field_magnitude.size(); ++i)
		{
			this->external_field_magnitude[i] = this->external_field_magnitude[i] * Constants::mu_B * mu_s[i];
		}

		// Generate Exchange neighbours
		Neighbours::Neighbours_from_Shells(*geometry, exchange_magnitude.size(), exchange_neighbours);
		// Generate DMI neighbours and normals
		Neighbours::Neighbours_from_Shells(*geometry, dmi_magnitude.size(), dmi_neighbours);
		// Generate DDI neighbours, magnitudes and normals
		// Create_Dipole_Neighbours();

		this->Update_Energy_Contributions();
	}
	
	void Hamiltonian_Heisenberg_Neighbours::Update_N_Neighbour_Shells(int n_shells_exchange, int n_shells_dmi)
	{
		if (this->exchange_magnitude.size() != n_shells_exchange)
		{
			this->exchange_magnitude = scalarfield(n_shells_exchange);
			// Re-calculate exchange neighbour list
		}
		if (this->dmi_magnitude.size() != n_shells_dmi)
		{
			this->dmi_magnitude = scalarfield(n_shells_dmi);
			// Re-calculate dmi neighbour list
		}
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
			this->idx_dd = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_dd = -1;
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

		// neighbours
		// Exchange
		if (this->idx_exchange >=0 )   E_Exchange(spins,energy_contributions_per_spin[idx_exchange].second);
		// DMI
		if (this->idx_dmi >=0 )        E_DMI(spins, energy_contributions_per_spin[idx_dmi].second);
		// DD
		if (this->idx_dd >=0 )         E_DD(spins, energy_contributions_per_spin[idx_dd].second);

		// Return
		//return this->E;
	}

	void Hamiltonian_Heisenberg_Neighbours::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int i = 0; i < this->external_field_index.size(); ++i)
		{
			Energy[external_field_index[i]] -= this->external_field_magnitude[i] * this->external_field_normal[i].dot(spins[external_field_index[i]]);
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::E_Anisotropy(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int i = 0; i < this->anisotropy_index.size(); ++i)
		{
			Energy[anisotropy_index[i]] -= this->anisotropy_magnitude[i] * std::pow(anisotropy_normal[i].dot(spins[anisotropy_index[i]]), 2.0);
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::E_Exchange(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			auto translations = translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
			for (unsigned int ineigh = 0; ineigh < exchange_neighbours.size(); ++ineigh)
			{
				if ( boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, exchange_neighbours[ineigh].translations) )
				{
					int jspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, exchange_neighbours[ineigh].translations);
					Energy[ispin] -= 0.5 * exchange_magnitude[exchange_neighbours[ineigh].idx_shell] * spins[ispin].dot(spins[jspin]);
				}
			}
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::E_DMI(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			auto translations = translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
			for (unsigned int ineigh = 0; ineigh < dmi_neighbours.size(); ++ineigh)
			{
				if ( boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, dmi_neighbours[ineigh].translations) )
				{
					int jspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, dmi_neighbours[ineigh].translations);
					Energy[ispin] -= 0.5 * dmi_magnitude[dmi_neighbours[ineigh].idx_shell] * dmi_normal[dmi_neighbours[ineigh].idx_shell].dot(spins[ispin].cross(spins[jspin]));
				}
			}
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::E_DD(const vectorfield & spins, scalarfield & Energy)
	{
		//scalar mult = -Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		scalar mult = 0.5*0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		scalar result = 0.0;

		for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
		{
			if (ddi_magnitude[ineigh] > 0.0)
			{
				// Energy[neighbours[ineigh][0]] -= mult / std::pow(ddi_magnitude[ineigh], 3.0) *
				// 	(3 * spins[neighbours[ineigh][1]].dot(DD_normal[ineigh]) * spins[neighbours[ineigh][0]].dot(DD_normal[ineigh]) - spins[neighbours[ineigh][0]].dot(spins[neighbours[ineigh][1]]));
				// Energy[neighbours[ineigh][1]] -= mult / std::pow(ddi_magnitude[ineigh], 3.0) *
				// 	(3 * spins[neighbours[ineigh][1]].dot(DD_normal[ineigh]) * spins[neighbours[ineigh][0]].dot(DD_normal[ineigh]) - spins[neighbours[ineigh][0]].dot(spins[neighbours[ineigh][1]]));
			}

		}
	}// end DipoleDipole



	void Hamiltonian_Heisenberg_Neighbours::Gradient(const vectorfield & spins, vectorfield & gradient)
	{
		// Set to zero
		Vectormath::fill(gradient, {0,0,0});

		// External field
		Gradient_Zeeman(gradient);

		// Anisotropy
		Gradient_Anisotropy(spins, gradient);

		// neighbours
		// Exchange
		this->Gradient_Exchange(spins, gradient);
		// DMI
		this->Gradient_DMI(spins, gradient);
		// DD
		this->Gradient_DD(spins, gradient);
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_Zeeman(vectorfield & gradient)
	{
		for (unsigned int i = 0; i < this->external_field_index.size(); ++i)
		{
			gradient[external_field_index[i]] -= this->external_field_magnitude[i] * this->external_field_normal[i];
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
	{
		for (unsigned int i = 0; i < this->anisotropy_index.size(); ++i)
		{
			gradient[anisotropy_index[i]] -= 2.0 * this->anisotropy_magnitude[i] * this->anisotropy_normal[i] * anisotropy_normal[i].dot(spins[anisotropy_index[i]]);
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			auto translations = translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
			for (unsigned int ineigh = 0; ineigh < exchange_neighbours.size(); ++ineigh)
			{
				if ( boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, exchange_neighbours[ineigh].translations) )
				{
					int jspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, exchange_neighbours[ineigh].translations);
					gradient[ispin] -= exchange_magnitude[exchange_neighbours[ineigh].idx_shell] * spins[jspin];
				}
			}
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			auto translations = translations_from_idx(geometry->n_cells, geometry->n_spins_basic_domain, ispin);
			for (unsigned int ineigh = 0; ineigh < dmi_neighbours.size(); ++ineigh)
			{
				if ( boundary_conditions_fulfilled(geometry->n_cells, boundary_conditions, translations, dmi_neighbours[ineigh].translations) )
				{
					int jspin = idx_from_translations(geometry->n_cells, geometry->n_spins_basic_domain, translations, dmi_neighbours[ineigh].translations);
					gradient[ispin] -= dmi_magnitude[dmi_neighbours[ineigh].idx_shell] * spins[jspin].cross(dmi_normal[dmi_neighbours[ineigh].idx_shell]);
				}
			}
		}
	}

	void Hamiltonian_Heisenberg_Neighbours::Gradient_DD(const vectorfield & spins, vectorfield & gradient)
	{
		//scalar mult = Constants::mu_B*Constants::mu_B*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		
		for (unsigned int ineigh = 0; ineigh < ddi_neighbours.size(); ++ineigh)
		{
			if (ddi_magnitude[ineigh] > 0.0)
			{
				scalar skalar_contrib = mult / std::pow(ddi_magnitude[ineigh], 3.0);
				// gradient[indices[ineigh][0]] -= skalar_contrib * (3 * DD_normal[ineigh] * spins[indices[ineigh][1]].dot(DD_normal[ineigh]) - spins[indices[ineigh][1]]);
				// gradient[indices[ineigh][1]] -= skalar_contrib * (3 * DD_normal[ineigh] * spins[indices[ineigh][0]].dot(DD_normal[ineigh]) - spins[indices[ineigh][0]]);
			}
		}
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
		//  		//		Loop over neighbours of this periodicity
		//  		// Exchange
		//  		for (unsigned int i_pair = 0; i_pair < this->exchange_neighbours.size(); ++i_pair)
		//  		{
		//  			for (int alpha = 0; alpha < 3; ++alpha)
		//  			{
		//  				int idx_i = 3*exchange_neighbours[i_pair][0] + alpha;
		//  				int idx_j = 3*exchange_neighbours[i_pair][1] + alpha;
		//  				hessian(idx_i,idx_j) += -exchange_magnitude[i_pair];
		//  				hessian(idx_j,idx_i) += -exchange_magnitude[i_pair];
		//  			}
		//  		}
		//  		// DMI
		//  		for (unsigned int i_pair = 0; i_pair < this->dmi_neighbours[i_periodicity].size(); ++i_pair)
		//  		{
		//  			for (int alpha = 0; alpha < 3; ++alpha)
		//  			{
		//  				for (int beta = 0; beta < 3; ++beta)
		//  				{
		//  					int idx_i = 3*dmi_neighbours[i_periodicity][i_pair][0] + alpha;
		//  					int idx_j = 3*dmi_neighbours[i_periodicity][i_pair][1] + beta;
		//  					if ( (alpha == 0 && beta == 1) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							-dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][2];
		//  						hessian(idx_j,idx_i) +=
		//  							-dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][2];
		//  					}
		//  					else if ( (alpha == 1 && beta == 0) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][2];
		//  						hessian(idx_j,idx_i) +=
		//  							dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][2];
		//  					}
		//  					else if ( (alpha == 0 && beta == 2) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][1];
		//  						hessian(idx_j,idx_i) +=
		//  							dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][1];
		//  					}
		//  					else if ( (alpha == 2 && beta == 0) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							-dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][1];
		//  						hessian(idx_j,idx_i) +=
		//  							-dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][1];
		//  					}
		//  					else if ( (alpha == 1 && beta == 2) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							-dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][0];
		//  						hessian(idx_j,idx_i) +=
		//  							-dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][0];
		//  					}
		//  					else if ( (alpha == 2 && beta == 1) )
		//  					{
		//  						hessian(idx_i,idx_j) +=
		//  							dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][0];
		//  						hessian(idx_j,idx_i) +=
		//  							dmi_magnitude[i_periodicity][i_pair] * dmi_normal[i_periodicity][i_pair][0];
		//  					}
		//  				}
		//  			}
		//  		}
		//  //		// Dipole-Dipole
		//  //		for (unsigned int i_pair = 0; i_pair < this->ddi_neighbours[i_periodicity].size(); ++i_pair)
		//  //		{
		//  //			// indices
		//  //			int idx_1 = ddi_neighbours[i_periodicity][i_pair][0];
		//  //			int idx_2 = ddi_neighbours[i_periodicity][i_pair][1];
		//  //			// prefactor
		//  //			scalar prefactor = 0.0536814951168
		//  //				* this->mu_s[idx_1] * this->mu_s[idx_2]
		//  //				/ std::pow(ddi_magnitude[i_periodicity][i_pair], 3);
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