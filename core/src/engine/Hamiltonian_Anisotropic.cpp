#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Dense>

#include <engine/Hamiltonian_Anisotropic.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>

using std::vector;
using std::function;

using namespace Data;

namespace Engine
{
	Hamiltonian_Anisotropic::Hamiltonian_Anisotropic(
		std::vector<scalar> mu_s,
		std::vector<int> external_field_index, std::vector<scalar> external_field_magnitude, std::vector<Vector3> external_field_normal,
		std::vector<int> anisotropy_index, std::vector<scalar> anisotropy_magnitude, std::vector<Vector3> anisotropy_normal,
		std::vector<std::vector<std::vector<int>>> Exchange_indices, std::vector<std::vector<scalar>> Exchange_magnitude,
		std::vector<std::vector<std::vector<int>>> DMI_indices, std::vector<std::vector<scalar>> DMI_magnitude, std::vector<std::vector<Vector3>> DMI_normal,
		std::vector<std::vector<std::vector<int>>> BQC_indices, std::vector<std::vector<scalar>> BQC_magnitude,
		std::vector<std::vector<std::vector<int>>> DD_indices, std::vector<std::vector<scalar>> DD_magnitude, std::vector<std::vector<Vector3>> DD_normal,
		std::vector<bool> boundary_conditions
	) :
		Hamiltonian(boundary_conditions),
		mu_s(mu_s),
		external_field_index(external_field_index), external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
		anisotropy_index(anisotropy_index), anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
		Exchange_indices(Exchange_indices), Exchange_magnitude(Exchange_magnitude),
		DMI_indices(DMI_indices), DMI_magnitude(DMI_magnitude), DMI_normal(DMI_normal),
		BQC_indices(BQC_indices), BQC_magnitude(BQC_magnitude),
		DD_indices(DD_indices), DD_magnitude(DD_magnitude), DD_normal(DD_normal)
	{
		// Renormalize the external field from Tesla to whatever
		for (unsigned int i = 0; i < external_field_magnitude.size(); ++i)
		{
			this->external_field_magnitude[i] = this->external_field_magnitude[i] * Vectormath::MuB() * mu_s[i];
		}
	}

	scalar Hamiltonian_Anisotropic::Energy(const std::vector<Vector3> & spins)
	{
		scalar sum = 0;
		auto e = Energy_Array(spins);
		for (auto E : e) sum += E;
		return sum;
	}

	std::vector<scalar> Hamiltonian_Anisotropic::Energy_Array(const std::vector<Vector3> & spins)
	{
		//     0           1           2      3    4     5       6
		// ext. field; anisotropy; exchange; dmi; bqc; 4spin; dipole-dipole
		std::vector<scalar> E(7, 0); // initialized with zeros
		//int nos = spins.size() / 3;

		// External field
		E_Zeeman(spins, E);

		// Anisotropy
		E_Anisotropy(spins, E);

		// Pairs
		//		Loop over periodicity
		for (int i_periodicity = 0; i_periodicity < 8; ++i_periodicity)
		{
			// Check if boundary conditions contain this periodicity
			if ((i_periodicity == 0)
				|| (i_periodicity == 1 && this->boundary_conditions[0])
				|| (i_periodicity == 2 && this->boundary_conditions[1])
				|| (i_periodicity == 3 && this->boundary_conditions[2])
				|| (i_periodicity == 4 && this->boundary_conditions[0] && this->boundary_conditions[1])
				|| (i_periodicity == 5 && this->boundary_conditions[0] && this->boundary_conditions[2])
				|| (i_periodicity == 6 && this->boundary_conditions[1] && this->boundary_conditions[2])
				|| (i_periodicity == 7 && this->boundary_conditions[0] && this->boundary_conditions[1] && this->boundary_conditions[2]))
			{
				//		Energies of this periodicity
				// Exchange
				this->E_Exchange(spins, Exchange_indices[i_periodicity], Exchange_magnitude[i_periodicity], E);
				// DMI
				this->E_DMI(spins, DMI_indices[i_periodicity], DMI_magnitude[i_periodicity], DMI_normal[i_periodicity], E);
				// BQC
				this->E_BQC(spins, BQC_indices[i_periodicity], BQC_magnitude[i_periodicity], E);
				// DD
				this->E_DD(spins, DD_indices[i_periodicity], DD_magnitude[i_periodicity], DD_normal[i_periodicity], E);
			}
		}

		// Triplet Interactions

		// Quadruplet Interactions

		// Return
		return E;
	}

	//std::vector<std::vector<scalar>> Hamiltonian_Anisotropic::Energy_Array_per_Spin(std::vector<scalar> & spins)
	//{
	//	int nos = spins.size() / 3;
	//	//     0           1           2      3    4   6     7
	//	// ext. field; anisotropy; exchange; dmi; bqc; 4spin; dipole-dipole
	//	std::vector<std::vector<scalar>> E(nos, std::vector<scalar>(7, 0)); // [nos][6], initialized with zeros
	//	std::vector<scalar> E_temp(7, 0);

	//	//// Loop over Spins
	//	//for (int i = 0; i<nos; ++i)
	//	//{
	//	//	// AT SOME POINT WE MIGHT CONSTRUCT A CLASS ANALOGOUS TO Spin_Pair FOR THIS
	//	//	// External field
	//	//	E_Zeeman(nos, spins, i, E[i]);
	//	//	// Anisotropy
	//	//	E_Anisotropy(nos, spins, i, E[i]);

	//	//	E[i][6] += E_DipoleDipole(nos, spins, i);
	//	//}

	//	//// Loop over Pairs
	//	//for (unsigned int i_pair = 0; i_pair<this->pairs.size(); ++i_pair)
	//	//{
	//	//	Data::Spin_Pair pair = this->pairs[i_pair];
	//	//	// loop over contributions in pair
	//	//	for (unsigned int j = 0; j<pair.energy_calls.size(); ++j)
	//	//	{
	//	//		for (unsigned int i = 0; i < E_temp.size(); ++i)
	//	//		{
	//	//			E_temp[i] = 0;
	//	//		}
	//	//		pair.energy_calls[j](this, nos, spins, pair, E_temp);
	//	//		for (unsigned int i = 0; i < E_temp.size(); ++i)
	//	//		{
	//	//			E[pair.idx_1][i] += 0.5*E_temp[i];
	//	//			E[pair.idx_2][i] += 0.5*E_temp[i];
	//	//		}
	//	//	}
	//	//}

	//	//// Triplet Interactions

	//	//// Quadruplet Interactions

	//	// Return
	//	return E;
	//}

	void Hamiltonian_Anisotropic::E_Zeeman(const std::vector<Vector3> & spins, std::vector<scalar> & Energy)
	{
		for (unsigned int i = 0; i < this->external_field_index.size(); ++i)
		{
			Energy[ENERGY_POS_ZEEMAN] -= this->external_field_magnitude[i] * this->external_field_normal[i].dot(spins[external_field_index[i]]);
		}
	}

	void Hamiltonian_Anisotropic::E_Anisotropy(const std::vector<Vector3> & spins, std::vector<scalar> & Energy)
	{
		for (unsigned int i = 0; i < this->anisotropy_index.size(); ++i)
		{
			Energy[ENERGY_POS_ANISOTROPY] -= this->anisotropy_magnitude[i] * std::pow(anisotropy_normal[i].dot(spins[anisotropy_index[i]]), 2.0);
		}
	}

	void Hamiltonian_Anisotropic::E_Exchange(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & J_ij, std::vector<scalar> & Energy)
	{
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			Energy[ENERGY_POS_EXCHANGE] -= J_ij[i_pair] * spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]]);;
		}
	}

	void Hamiltonian_Anisotropic::E_DMI(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & DMI_magnitude, std::vector<Vector3> & DMI_normal, std::vector<scalar> & Energy)
	{
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			Energy[ENERGY_POS_DMI] -= DMI_magnitude[i_pair] * DMI_normal[i_pair].dot(spins[indices[i_pair][0]].cross(spins[indices[i_pair][1]]));
		}
	}


	void Hamiltonian_Anisotropic::E_BQC(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & B_ij, std::vector<scalar> & Energy)
	{
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			Energy[ENERGY_POS_BQC] -= B_ij[i_pair] * spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]]);
		}
	}

	void Hamiltonian_Anisotropic::E_DD(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & DD_magnitude, std::vector<Vector3> & DD_normal, std::vector<scalar> & Energy)
	{
		//scalar mult = -Utility::Vectormath::MuB()*Utility::Vectormath::MuB()*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		scalar result = 0.0;

		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			if (DD_magnitude[i_pair] > 0.0)
			{
				Energy[ENERGY_POS_DD] -= mult / std::pow(DD_magnitude[i_pair], 3.0) *
					(3 * spins[indices[i_pair][1]].dot(DD_normal[i_pair]) * spins[indices[i_pair][0]].dot(DD_normal[i_pair]) - spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]]));
			}

		}
	}// end DipoleDipole


	void Hamiltonian_Anisotropic::Effective_Field(const std::vector<Vector3> & spins, std::vector<Vector3> & field)
	{
		int nos = spins.size();
		// Loop over Spins
		for (int i = 0; i < nos; ++i)
		{
			field[i] = { 0,0,0 };
		}

		// External field
		Field_Zeeman(spins, field);

		// Anisotropy
		Field_Anisotropy(spins, field);

		// Pairs
		//		Loop over periodicity
		for (int i_periodicity = 0; i_periodicity < 8; ++i_periodicity)
		{
			// Check if boundary conditions contain this periodicity
			if ((i_periodicity == 0)
				|| (i_periodicity == 1 && this->boundary_conditions[0])
				|| (i_periodicity == 2 && this->boundary_conditions[1])
				|| (i_periodicity == 3 && this->boundary_conditions[2])
				|| (i_periodicity == 4 && this->boundary_conditions[0] && this->boundary_conditions[1])
				|| (i_periodicity == 5 && this->boundary_conditions[0] && this->boundary_conditions[2])
				|| (i_periodicity == 6 && this->boundary_conditions[1] && this->boundary_conditions[2])
				|| (i_periodicity == 7 && this->boundary_conditions[0] && this->boundary_conditions[1] && this->boundary_conditions[2]))
			{
				//		Fields of this periodicity
				// Exchange
				this->Field_Exchange(spins, Exchange_indices[i_periodicity], Exchange_magnitude[i_periodicity], field);
				// DMI
				this->Field_DMI(spins, DMI_indices[i_periodicity], DMI_magnitude[i_periodicity], DMI_normal[i_periodicity], field);
				// BQC
				this->Field_BQC(spins, BQC_indices[i_periodicity], BQC_magnitude[i_periodicity], field);
				// DD
				this->Field_DD(spins, DD_indices[i_periodicity], DD_magnitude[i_periodicity], DD_normal[i_periodicity], field);
			}
		}

		// Triplet Interactions

		// Quadruplet Interactions
	}

	void Hamiltonian_Anisotropic::Field_Zeeman(const std::vector<Vector3> & spins, std::vector<Vector3> & eff_field)
	{
		for (unsigned int i = 0; i < this->external_field_index.size(); ++i)
		{
			eff_field[external_field_index[i]] += this->external_field_magnitude[i] * this->external_field_normal[i];
		}
	}

	void Hamiltonian_Anisotropic::Field_Anisotropy(const std::vector<Vector3> & spins, std::vector<Vector3> & eff_field)
	{
		for (unsigned int i = 0; i < this->anisotropy_index.size(); ++i)
		{
			eff_field[anisotropy_index[i]] += 2.0 * this->anisotropy_magnitude[i] * this->anisotropy_normal[i] * anisotropy_normal[i].dot(spins[anisotropy_index[i]]);
		}
	}

	void Hamiltonian_Anisotropic::Field_Exchange(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & J_ij, std::vector<Vector3> & eff_field)
	{
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			eff_field[indices[i_pair][0]] += J_ij[i_pair] * spins[indices[i_pair][1]];
			eff_field[indices[i_pair][1]] += J_ij[i_pair] * spins[indices[i_pair][0]];
		}
	}

	void Hamiltonian_Anisotropic::Field_DMI(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & DMI_magnitude, std::vector<Vector3> & DMI_normal, std::vector<Vector3> & eff_field)
	{
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			eff_field[indices[i_pair][0]] += DMI_magnitude[i_pair] * spins[indices[i_pair][1]].cross(DMI_normal[i_pair]);
			eff_field[indices[i_pair][1]] -= DMI_magnitude[i_pair] * spins[indices[i_pair][0]].cross(DMI_normal[i_pair]);
		}
	}

	void Hamiltonian_Anisotropic::Field_BQC(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & B_ij, std::vector<Vector3> & eff_field)
	{
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			eff_field[indices[i_pair][0]] += 2 * B_ij[i_pair] * spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]]) * spins[indices[i_pair][1]];
			eff_field[indices[i_pair][1]] += 2 * B_ij[i_pair] * spins[indices[i_pair][0]].dot(spins[indices[i_pair][1]]) * spins[indices[i_pair][0]];
		}
	}
	void Hamiltonian_Anisotropic::Field_DD(const std::vector<Vector3> & spins, std::vector<std::vector<int>> & indices, std::vector<scalar> & DD_magnitude, std::vector<Vector3> & DD_normal, std::vector<Vector3> & eff_field)
	{
		//scalar mult = Utility::Vectormath::MuB()*Utility::Vectormath::MuB()*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		scalar mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		
		for (unsigned int i_pair = 0; i_pair < indices.size(); ++i_pair)
		{
			if (DD_magnitude[i_pair] > 0.0)
			{
				scalar skalar_contrib = mult / std::pow(DD_magnitude[i_pair], 3.0);
				eff_field[indices[i_pair][0]] += skalar_contrib * (3 * DD_normal[i_pair] * spins[indices[i_pair][1]].dot(DD_normal[i_pair]) - spins[indices[i_pair][1]]);
				eff_field[indices[i_pair][1]] += skalar_contrib * (3 * DD_normal[i_pair] * spins[indices[i_pair][0]].dot(DD_normal[i_pair]) - spins[indices[i_pair][0]]);
			}
		}
	}//end Field_DipoleDipole

	void Hamiltonian_Anisotropic::Hessian(const std::vector<Vector3> & spins, MatrixX & hessian)
	{
		//int nos = spins.size() / 3;

		//// Set to zero
		//for (auto& h : hessian) h = 0;

		//// Single Spin elements
		//for (int alpha = 0; alpha < 3; ++alpha)
		//{
		//	for (unsigned int i = 0; i < anisotropy_index.size(); ++i)
		//	{
		//		int idx = anisotropy_index[i];
		//		scalar x = -2.0*this->anisotropy_magnitude[i] * std::pow(this->anisotropy_normal[i][alpha], 2);
		//		hessian[idx + alpha*nos + 3 * nos*(idx + alpha*nos)] += -2.0*this->anisotropy_magnitude[i]*std::pow(this->anisotropy_normal[i][alpha],2);
		//	}
		//}

		//// Spin Pair elements
		//for (int i_periodicity = 0; i_periodicity < 8; ++i_periodicity)
		//{
		//	//		Check if boundary conditions contain this periodicity
		//	if ((i_periodicity == 0)
		//		|| (i_periodicity == 1 && this->boundary_conditions[0])
		//		|| (i_periodicity == 2 && this->boundary_conditions[1])
		//		|| (i_periodicity == 3 && this->boundary_conditions[2])
		//		|| (i_periodicity == 4 && this->boundary_conditions[0] && this->boundary_conditions[1])
		//		|| (i_periodicity == 5 && this->boundary_conditions[0] && this->boundary_conditions[2])
		//		|| (i_periodicity == 6 && this->boundary_conditions[1] && this->boundary_conditions[2])
		//		|| (i_periodicity == 7 && this->boundary_conditions[0] && this->boundary_conditions[1] && this->boundary_conditions[2]))
		//	{
		//		//		Loop over pairs of this periodicity
		//		// Exchange
		//		for (unsigned int i_pair = 0; i_pair < this->Exchange_indices[i_periodicity].size(); ++i_pair)
		//		{
		//			for (int alpha = 0; alpha < 3; ++alpha)
		//			{
		//				int idx_h = Exchange_indices[i_periodicity][i_pair][0] + alpha*nos + 3 * nos*(Exchange_indices[i_periodicity][i_pair][1] + alpha*nos);
		//				hessian[idx_h] += -Exchange_magnitude[i_periodicity][i_pair];
		//			}
		//		}
		//		// DMI
		//		for (unsigned int i_pair = 0; i_pair < this->DMI_indices[i_periodicity].size(); ++i_pair)
		//		{
		//			for (int alpha = 0; alpha < 3; ++alpha)
		//			{
		//				for (int beta = 0; beta < 3; ++beta)
		//				{
		//					int idx_h = DMI_indices[i_periodicity][i_pair][0] + alpha*nos + 3 * nos*(DMI_indices[i_periodicity][i_pair][1] + beta*nos);
		//					if ( (alpha == 0 && beta == 1) || (alpha == 1 && beta == 0) )
		//					{
		//						hessian[idx_h] +=
		//							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][2];
		//					}
		//					else if ( (alpha == 0 && beta == 2) || (alpha == 2 && beta == 0) )
		//					{
		//						hessian[idx_h] +=
		//							-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][1];
		//					}
		//					else if ( (alpha == 1 && beta == 2) || (alpha == 2 && beta == 1) )
		//					{
		//						hessian[idx_h] +=
		//							DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][0];
		//					}
		//				}
		//			}
		//		}
		//		// Dipole-Dipole
		//		for (unsigned int i_pair = 0; i_pair < this->DD_indices[i_periodicity].size(); ++i_pair)
		//		{
		//			// indices
		//			int idx_1 = DD_indices[i_periodicity][i_pair][0];
		//			int idx_2 = DD_indices[i_periodicity][i_pair][1];
		//			// prefactor
		//			scalar prefactor = 0.0536814951168
		//				* this->mu_s[idx_1] * this->mu_s[idx_2]
		//				/ std::pow(DD_magnitude[i_periodicity][i_pair], 3);
		//			// components
		//			for (int alpha = 0; alpha < 3; ++alpha)
		//			{
		//				for (int beta = 0; beta < 3; ++beta)
		//				{
		//					int idx_h = idx_1 + alpha*nos + 3 * nos*(idx_2 + beta*nos);
		//					if (alpha == beta)
		//						hessian[idx_h] += prefactor;
		//					hessian[idx_h] += -3.0*prefactor*DD_normal[i_periodicity][i_pair][alpha] * DD_normal[i_periodicity][i_pair][beta];
		//				}
		//			}
		//		}
		//	}// end if periodicity
		//}// end for periodicity
	}

	// Hamiltonian name as string
	static const std::string name = "Anisotropic Heisenberg";
	const std::string& Hamiltonian_Anisotropic::Name() { return name; }
}
