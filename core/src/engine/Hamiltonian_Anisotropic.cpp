#define _USE_MATH_DEFINES
#include <cmath>

#include <engine/Hamiltonian_Anisotropic.hpp>

#include "Vectormath.hpp"
#include <Spin_System.hpp>
#include "Vectoroperators.hpp"
#include "Vectormath.hpp"
#include "Neighbours.hpp"

using std::vector;
using std::function;

using namespace Data;

namespace Engine
{
	Hamiltonian_Anisotropic::Hamiltonian_Anisotropic(
		std::vector<double> mu_s,
		std::vector<double> external_field_magnitude, std::vector<std::vector<double>> external_field_normal,
		std::vector<int> anisotropy_index, std::vector<double> anisotropy_magnitude, std::vector<std::vector<double>> anisotropy_normal,
		std::vector<std::vector<std::vector<int>>> Exchange_indices, std::vector<std::vector<double>> Exchange_magnitude,
		std::vector<std::vector<std::vector<int>>> DMI_indices, std::vector<std::vector<double>> DMI_magnitude, std::vector<std::vector<std::vector<double>>> DMI_normal,
		std::vector<std::vector<std::vector<int>>> BQC_indices, std::vector<std::vector<double>> BQC_magnitude,
		std::vector<std::vector<std::vector<int>>> DD_indices, std::vector<std::vector<double>> DD_magnitude, std::vector<std::vector<std::vector<double>>> DD_normal,
		std::vector<bool> boundary_conditions
	) :
		Hamiltonian(boundary_conditions),
		mu_s(mu_s),
		external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
		anisotropy_index(anisotropy_index), anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
		Exchange_indices(Exchange_indices), Exchange_magnitude(Exchange_magnitude),
		DMI_indices(DMI_indices), DMI_magnitude(DMI_magnitude), DMI_normal(DMI_normal),
		BQC_indices(BQC_indices), BQC_magnitude(BQC_magnitude),
		DD_indices(DD_indices), DD_magnitude(DD_magnitude), DD_normal(DD_normal)
	{
		// Renormalize the external field from Tesla to whatever
		for (unsigned int i = 0; i < external_field_magnitude.size(); ++i)
		{
			this->external_field_magnitude[i] = this->external_field_magnitude[i] * Utility::Vectormath::MuB() * mu_s[i];
		}
	}

	double Hamiltonian_Anisotropic::Energy(std::vector<double> & spins)
	{
		return sum(Energy_Array(spins));
	}

	std::vector<double> Hamiltonian_Anisotropic::Energy_Array(std::vector<double> & spins)
	{
		//     0           1           2      3    4     5       6
		// ext. field; anisotropy; exchange; dmi; bqc; 4spin; dipole-dipole
		std::vector<double> E(7, 0); // initialized with zeros
		int nos = spins.size() / 3;

		// Loop over Spins
		for (int i = 0; i<nos; ++i)
		{
			// External field
			E_Zeeman(nos, spins, i, E);
		}

		// Anisotropy
		E_Anisotropy(nos, spins, E);

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
				//		Loop over pairs of this periodicity
				// Exchange
				for (unsigned int i_pair = 0; i_pair < this->Exchange_indices[i_periodicity].size(); ++i_pair)
				{
					this->E_Exchange(nos, spins, Exchange_indices[i_periodicity][i_pair], Exchange_magnitude[i_periodicity][i_pair], E);
				}
				// DMI
				for (unsigned int i_pair = 0; i_pair < this->DMI_indices[i_periodicity].size(); ++i_pair)
				{
					this->E_DMI(nos, spins, DMI_indices[i_periodicity][i_pair], DMI_magnitude[i_periodicity][i_pair], DMI_normal[i_periodicity][i_pair], E);
				}
				// BQC
				for (unsigned int i_pair = 0; i_pair < this->BQC_indices[i_periodicity].size(); ++i_pair)
				{
					this->E_BQC(nos, spins, BQC_indices[i_periodicity][i_pair], BQC_magnitude[i_periodicity][i_pair], E);
				}
				// DD
				for (unsigned int i_pair = 0; i_pair < this->DD_indices[i_periodicity].size(); ++i_pair)
				{
					this->E_DD(nos, spins, DD_indices[i_periodicity][i_pair], DD_magnitude[i_periodicity][i_pair], DD_normal[i_periodicity][i_pair], E);
				}
			}
		}

		// Triplet Interactions

		// Quadruplet Interactions

		// Return
		return E;
	}

	//std::vector<std::vector<double>> Hamiltonian_Anisotropic::Energy_Array_per_Spin(std::vector<double> & spins)
	//{
	//	int nos = spins.size() / 3;
	//	//     0           1           2      3    4   6     7
	//	// ext. field; anisotropy; exchange; dmi; bqc; 4spin; dipole-dipole
	//	std::vector<std::vector<double>> E(nos, std::vector<double>(7, 0)); // [nos][6], initialized with zeros
	//	std::vector<double> E_temp(7, 0);

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

	void Hamiltonian_Anisotropic::E_Zeeman(int nos, std::vector<double> & spins, int ispin, std::vector<double> & Energy)
	{
		for (int i = 0; i < 3; ++i)
		{
			Energy[ENERGY_POS_ZEEMAN] -= this->external_field_magnitude[ispin] * this->external_field_normal[i][ispin] * spins[ispin + i*nos];
		}
	}

	void Hamiltonian_Anisotropic::E_Anisotropy(int nos, std::vector<double> & spins, std::vector<double> & Energy)
	{
		double t = 0;
		for (unsigned int i = 0; i < this->anisotropy_index.size(); ++i)
		{
			t = 0;
			int index = this->anisotropy_index[i];
			for (int dim = 0; dim < 3; ++dim)
			{
				t += this->anisotropy_normal[i][dim] * spins[index + dim*nos];
			}
			Energy[ENERGY_POS_ANISOTROPY] -= this->anisotropy_magnitude[i] * std::pow(t, 2.0);
		}
	}

	void Hamiltonian_Anisotropic::E_Exchange(int nos, std::vector<double> & spins, std::vector<int> & indices, double J_ij, std::vector<double> & Energy)
	{
		double ss = 0;
		for (int i = 0; i < 3; ++i)
		{
			ss += spins[indices[0] + i*nos] * spins[indices[1] + i*nos];
		}
		Energy[ENERGY_POS_EXCHANGE] -= J_ij * ss;
	}

	void Hamiltonian_Anisotropic::E_DMI(int nos, std::vector<double> & spins, std::vector<int> & indices, double & DMI_magnitude, std::vector<double> & DMI_normal, std::vector<double> & Energy)
	{
		std::vector<double> cross(3);
		for (int dim = 0; dim < 3; ++dim)
		{
			cross[dim] = spins[((dim + 1) % 3)*nos + indices[0]] * spins[((dim + 2) % 3)*nos + indices[1]]
				- spins[((dim + 2) % 3)*nos + indices[0]] * spins[((dim + 1) % 3)*nos + indices[1]];
		}
		/*for (int i = 0; i < 3; ++i)
		{
		cross[0] = s.spins[pair.idx_1 + s.nos] * s.spins[pair.idx_2 + 2*s.nos] - s.spins[pair.idx_1 + 2*s.nos] * s.spins[pair.idx_2 + s.nos];
		cross[1] = s.spins[pair.idx_1 + 2*s.nos] * s.spins[pair.idx_2] - s.spins[pair.idx_1] * s.spins[pair.idx_2 + 2*s.nos];
		cross[2] = s.spins[pair.idx_1] * s.spins[pair.idx_2 + s.nos] - s.spins[pair.idx_1 + s.nos] * s.spins[pair.idx_2];
		}*/
		for (int i = 0; i < 3; ++i)
		{
			Energy[ENERGY_POS_DMI] -= DMI_magnitude * DMI_normal[i] * cross[i];
		}
	}


	void Hamiltonian_Anisotropic::E_BQC(int nos, std::vector<double> & spins, std::vector<int> & indices, double B_ij, std::vector<double> & Energy)
	{
		for (int i = 0; i < 3; ++i)
		{
			Energy[ENERGY_POS_BQC] -= B_ij * spins[indices[0] + i*nos] * spins[indices[1] + i*nos];
		}
	}

	void Hamiltonian_Anisotropic::E_DD(int nos, std::vector<double> & spins, std::vector<int> & indices, double & DD_magnitude, std::vector<double> & DD_normal, std::vector<double> & Energy)
	{
		//double mult = -Utility::Vectormath::MuB()*Utility::Vectormath::MuB()*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		double mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		double result = 0.0;

		Energy[ENERGY_POS_DD] -= mult * this->mu_s[indices[0]] * this->mu_s[indices[1]] / std::pow(DD_magnitude, 3.0) *
			(3  *   ( spins[indices[1]]           * DD_normal[0]
					+ spins[indices[1] + 1 * nos] * DD_normal[1]
					+ spins[indices[1] + 2 * nos] * DD_normal[2])
			   	*   ( spins[indices[0]]           * DD_normal[0]
					+ spins[indices[0] + 1 * nos] * DD_normal[1]
					+ spins[indices[0] + 2 * nos] * DD_normal[2])
				-   ( spins[indices[0]]           * spins[indices[1]]
					+ spins[indices[0] + 1 * nos] * spins[indices[1] + 1 * nos]
					+ spins[indices[0] + 2 * nos] * spins[indices[1] + 2 * nos]));
	}// end DipoleDipole


	void Hamiltonian_Anisotropic::Effective_Field(const std::vector<double> & spins, std::vector<double> & field)
	{
		int nos = spins.size()/3;
		// Loop over Spins
		for (int i = 0; i < nos; ++i)
		{
			field[i] = 0; field[i + nos] = 0; field[i + 2 * nos] = 0;
			// AT SOME POINT WE MIGHT CONSTRUCT A CLASS ANALOGOUS TO Spin_Pair FOR THIS
			// External field
			Field_Zeeman(nos, spins, field, i);
		}

		// Anisotropy
		Field_Anisotropy(nos, spins, field);

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
				//		Loop over pairs of this periodicity
				// Exchange
				for (unsigned int i_pair = 0; i_pair < this->Exchange_indices[i_periodicity].size(); ++i_pair)
				{
					this->Field_Exchange(nos, spins, Exchange_indices[i_periodicity][i_pair], Exchange_magnitude[i_periodicity][i_pair], field);
				}
				// DMI
				for (unsigned int i_pair = 0; i_pair < this->DMI_indices[i_periodicity].size(); ++i_pair)
				{
					this->Field_DMI(nos, spins, DMI_indices[i_periodicity][i_pair], DMI_magnitude[i_periodicity][i_pair], DMI_normal[i_periodicity][i_pair], field);
				}
				// BQC
				for (unsigned int i_pair = 0; i_pair < this->BQC_indices[i_periodicity].size(); ++i_pair)
				{
					this->Field_BQC(nos, spins, BQC_indices[i_periodicity][i_pair], BQC_magnitude[i_periodicity][i_pair], field);
				}
				// DD
				for (unsigned int i_pair = 0; i_pair < this->DD_indices[i_periodicity].size(); ++i_pair)
				{
					this->Field_DD(nos, spins, DD_indices[i_periodicity][i_pair], DD_magnitude[i_periodicity][i_pair], DD_normal[i_periodicity][i_pair], field);
				}
			}
		}

		// Triplet Interactions

		// Quadruplet Interactions
	}

	void Hamiltonian_Anisotropic::Field_Zeeman(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		for (int i = 0; i < 3; ++i)
		{
			eff_field[ispin + i*nos] += this->external_field_magnitude[ispin] * this->external_field_normal[i][ispin];
		}
	}

	void Hamiltonian_Anisotropic::Field_Anisotropy(int nos, const std::vector<double> & spins, std::vector<double> & eff_field)
	{
		double t = 0;
		for (unsigned int i = 0; i < this->anisotropy_index.size(); ++i)
		{
			int index = this->anisotropy_index[i];
			t = 0;
			for (int dim = 0; dim < 3; ++dim)
			{
				t += this->anisotropy_normal[i][dim] * spins[index + dim*nos];
			}

			for (int dim = 0; dim < 3; ++dim)
			{
				eff_field[index + dim*nos] += 2.0 * this->anisotropy_magnitude[i] * this->anisotropy_normal[i][dim] * t;
			}
		}
	}

	void Hamiltonian_Anisotropic::Field_Exchange(int nos, const std::vector<double> & spins, std::vector<int> & indices, double J_ij, std::vector<double> & eff_field)
	{
		for (int i = 0; i < 3; ++i)
		{
			eff_field[indices[0] + i*nos] += J_ij * spins[indices[1] + i*nos];
			eff_field[indices[1] + i*nos] += J_ij * spins[indices[0] + i*nos];
		}
	}

	void Hamiltonian_Anisotropic::Field_DMI(int nos, const std::vector<double> & spins, std::vector<int> & indices, double & DMI_magnitude, std::vector<double> & DMI_normal, std::vector<double> & eff_field)
	{
		std::vector<double> cross1(3), cross2(3);
		for (int dim = 0; dim < 3; ++dim)
		{
			cross1[dim] = spins[((dim + 1) % 3)*nos + indices[1]] * DMI_normal[((dim + 2) % 3)]
				- spins[((dim + 2) % 3)*nos + indices[1]] * DMI_normal[((dim + 1) % 3)];
			cross2[dim] = -spins[((dim + 1) % 3)*nos + indices[0]] * DMI_normal[((dim + 2) % 3)]
				+ spins[((dim + 2) % 3)*nos + indices[0]] * DMI_normal[((dim + 1) % 3)];
		}

		//cross1[0] = s.spins[pair.idx_2 + s.nos] * pair.D_ij_normal[2] - s.spins[pair.idx_2 + 2*s.nos] * pair.D_ij_normal[1];
		//cross1[1] = s.spins[pair.idx_2 + 2*s.nos] * pair.D_ij_normal[0] - s.spins[pair.idx_2] * pair.D_ij_normal[2];
		//cross1[2] = s.spins[pair.idx_2] * pair.D_ij_normal[1] - s.spins[pair.idx_2 + s.nos] * pair.D_ij_normal[0];

		//cross2[0] = -s.spins[pair.idx_1 + s.nos] * pair.D_ij_normal[2] - s.spins[pair.idx_1 + 2 * s.nos] * pair.D_ij_normal[1];
		//cross2[1] = -s.spins[pair.idx_1 + 2 * s.nos] * pair.D_ij_normal[0] - s.spins[pair.idx_1] * pair.D_ij_normal[2];
		//cross2[2] = -s.spins[pair.idx_1] * pair.D_ij_normal[1] - s.spins[pair.idx_1 + s.nos] * pair.D_ij_normal[0];

		for (int i = 0; i < 3; ++i)
		{
			eff_field[indices[0] + i*nos] += DMI_magnitude * cross1[i];
			eff_field[indices[1] + i*nos] += DMI_magnitude * cross2[i];
		}
	}

	void Hamiltonian_Anisotropic::Field_BQC(int nos, const std::vector<double> & spins, std::vector<int> & indices, double B_ij, std::vector<double> & eff_field)
	{
		double ss = 0.0;
		for (int i = 0; i < 3; ++i)
		{
			ss += spins[indices[0] + i*nos] * spins[indices[1] + i*nos];
		}
		for (int i = 0; i < 3; ++i)
		{
			eff_field[indices[0] + i*nos] += 2.0 * B_ij * ss * spins[indices[1] + i*nos];
			eff_field[indices[1] + i*nos] += 2.0 * B_ij * ss * spins[indices[0] + i*nos];
		}
	}
	void Hamiltonian_Anisotropic::Field_DD(int nos, const std::vector<double> & spins, std::vector<int> & indices, double & DD_magnitude, std::vector<double> & DD_normal, std::vector<double> & eff_field)
	{
		eff_field[0] = 0.0; eff_field[1] = 0.0; eff_field[2] = 0.0;
		int dim;
		//double mult = Utility::Vectormath::MuB()*Utility::Vectormath::MuB()*1.0 / 4.0 / M_PI; // multiply with mu_B^2
		double mult = 0.0536814951168; // mu_0*mu_B**2/(4pi*10**-30) -- the translations are in angstr�m, so the |r|[m] becomes |r|[m]*10^-10
		double skalar_contrib, dotprod1, dotprod0;
		
		skalar_contrib = mult * this->mu_s[indices[0]] * this->mu_s[indices[1]] / std::pow(DD_magnitude, 3.0);
		dotprod1 = spins[indices[1]] * DD_normal[0]
			+ spins[1 * nos + indices[1]] * DD_normal[1]
			+ spins[2 * nos + indices[1]] * DD_normal[2];
		dotprod0 = spins[indices[0]] * DD_normal[0]
			+ spins[1 * nos + indices[0]] * DD_normal[1]
			+ spins[2 * nos + indices[0]] * DD_normal[2];
		for (dim = 0; dim < 3; ++dim)
		{
			eff_field[dim*nos + indices[0]] += skalar_contrib * (3 * dotprod1*DD_normal[dim] - spins[dim * nos + indices[1]]);
			eff_field[dim*nos + indices[1]] += skalar_contrib * (3 * dotprod0*DD_normal[dim] - spins[dim * nos + indices[0]]);
		}
	}//end Field_DipoleDipole

	void Hamiltonian_Anisotropic::Hessian(const std::vector<double> & spins, std::vector<double> & hessian)
	{
		int nos = spins.size() / 3;

		// Single Spin elements
		for (int alpha = 0; alpha < 3; ++alpha)
		{
			for (int i = 0; i < nos; ++i)
			{
				hessian[i + alpha*nos + 3 * nos*(i + alpha*nos)] = -2.0*this->anisotropy_magnitude[i]*std::pow(this->anisotropy_normal[alpha][i],2);
			}
		}

		// Spin Pair elements
		for (int i_periodicity = 0; i_periodicity < 8; ++i_periodicity)
		{
			//		Check if boundary conditions contain this periodicity
			if ((i_periodicity == 0)
				|| (i_periodicity == 1 && this->boundary_conditions[0])
				|| (i_periodicity == 2 && this->boundary_conditions[1])
				|| (i_periodicity == 3 && this->boundary_conditions[2])
				|| (i_periodicity == 4 && this->boundary_conditions[0] && this->boundary_conditions[1])
				|| (i_periodicity == 5 && this->boundary_conditions[0] && this->boundary_conditions[2])
				|| (i_periodicity == 6 && this->boundary_conditions[1] && this->boundary_conditions[2])
				|| (i_periodicity == 7 && this->boundary_conditions[0] && this->boundary_conditions[1] && this->boundary_conditions[2]))
			{
				//		Loop over pairs of this periodicity
				// Exchange
				for (unsigned int i_pair = 0; i_pair < this->Exchange_indices[i_periodicity].size(); ++i_pair)
				{
					for (int alpha = 0; alpha < 3; ++alpha)
					{
						int idx_h = Exchange_indices[i_periodicity][i_pair][0] + alpha*nos + 3 * nos*(Exchange_indices[i_periodicity][i_pair][1] + alpha*nos);
						hessian[idx_h] = -Exchange_magnitude[i_periodicity][i_pair];
					}
				}
				// DMI
				for (unsigned int i_pair = 0; i_pair < this->DMI_indices[i_periodicity].size(); ++i_pair)
				{
					for (int alpha = 0; alpha < 3; ++alpha)
					{
						for (int beta = 0; beta < 3; ++beta)
						{
							int idx_h = DMI_indices[i_periodicity][i_pair][0] + alpha*nos + 3 * nos*(DMI_indices[i_periodicity][i_pair][1] + beta*nos);
							if ( (alpha == 0 && beta == 1) || (alpha == 1 && beta == 0) )
							{
								hessian[idx_h] =
									DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][2];
							}
							else if ( (alpha == 0 && beta == 2) || (alpha == 2 && beta == 0) )
							{
								hessian[idx_h] =
									-DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][1];
							}
							else if ( (alpha == 1 && beta == 2) || (alpha == 2 && beta == 1) )
							{
								hessian[idx_h] =
									DMI_magnitude[i_periodicity][i_pair] * DMI_normal[i_periodicity][i_pair][0];
							}
						}
					}
				}
			}
		}
	}

	// Hamiltonian name as string
	std::string Hamiltonian_Anisotropic::Name() { return "Anisotropic Heisenberg"; }
}