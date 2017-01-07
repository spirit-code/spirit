#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include <engine/Hamiltonian_Isotropic.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <utility/Logging.hpp>

using namespace Utility;

namespace Engine
{
	Hamiltonian_Isotropic::Hamiltonian_Isotropic(
		std::vector<bool> boundary_conditions, scalar external_field_magnitude_i, Vector3 external_field_normal, scalar mu_s,
		scalar anisotropy_magnitude, Vector3 anisotropy_normal,
		int n_neigh_shells, std::vector<scalar> jij, scalar dij, scalar bij, scalar kijkl, scalar dd_radius,
		Data::Geometry geometry) :
		Hamiltonian(boundary_conditions),
		mu_s(mu_s),
		external_field_magnitude(external_field_magnitude_i), external_field_normal(external_field_normal),
		anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
		n_neigh_shells(n_neigh_shells), jij(jij), dij(dij), bij(bij), kijkl(kijkl), dd_radius(dd_radius)
	{
		// Rescale magnetic field from Tesla to meV
		external_field_magnitude = external_field_magnitude * Vectormath::MuB() * mu_s;
		external_field_normal.normalize();

		this->Update_Energy_Contributions();

		// Calculate Neighbours
		Log(Log_Level::Info, Log_Sender::All, "Building Neighbours ...");
		Engine::Neighbours::Create_Neighbours(geometry, boundary_conditions, n_neigh_shells,
			n_spins_in_shell, neigh, n_4spin, max_n_4spin, neigh_4spin, dm_normal, segments, segments_pos);
		Engine::Neighbours::Create_Dipole_Neighbours(geometry, boundary_conditions,
			dd_radius, dd_neigh, dd_neigh_pos, dd_normal, dd_distance);
		Log(Log_Level::Info, Log_Sender::All, "Done Caclulating Neighbours");
	}


	void Hamiltonian_Isotropic::Update_Energy_Contributions()
	{
		this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

		// External field
		if (this->external_field_magnitude != 0)
		{
			this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
			this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_zeeman = -1;
		// Anisotropy
		if (this->anisotropy_magnitude != 0)
		{
			this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0)});
			this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_anisotropy = -1;
		// Exchange
		for (auto _j : jij)
		{
			if (_j != 0)
			{
				this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0)});
				this->idx_exchange = this->energy_contributions_per_spin.size()-1;
				break;
			}
			else this->idx_exchange = -1;
		}
		// DMI
		if (dij != 0)
		{
			this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0)});
			this->idx_dmi = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_dmi = -1;
		// BQC
		if (bij != 0)
		{
			this->energy_contributions_per_spin.push_back({"BQC", scalarfield(0)});
			this->idx_bqc = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_bqc = -1;
		// FSC
		if (kijkl != 0)
		{
			this->energy_contributions_per_spin.push_back({"FSC", scalarfield(0)});
			this->idx_fsc = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_fsc = -1;
		// Dipole-Dipole
		if (this->dd_radius > 0)
		{
			this->energy_contributions_per_spin.push_back({"DD", scalarfield(0)});
			this->idx_dd = this->energy_contributions_per_spin.size()-1;
		}
		else this->idx_dd = -1;
	}



	void Hamiltonian_Isotropic::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
	{
		//========================= Init local vars ================================
		int nos = spins.size();
		int i = 0, istart = -1, istop = istart + 1;
		scalar f0 = 1.0;
		if (istart == -1) { istart = 0; istop = nos; f0 = 0.5; }
		//------------------------ End Init ----------------------------------------

		for (auto& pair : energy_contributions_per_spin)
		{
			// Allocate if not already allocated
			if (pair.second.size() != nos) pair.second = scalarfield(nos, 0);
			// Otherwise set to zero
			else for (auto& pair : energy_contributions_per_spin) Vectormath::fill(pair.second, 0);
		}

		if (idx_zeeman >= 0)     E_Zeeman(spins, energy_contributions_per_spin[idx_zeeman].second);
		if (idx_exchange >= 0)   E_Exchange(spins, energy_contributions_per_spin[idx_exchange].second);
		if (idx_anisotropy >= 0) E_Anisotropic(spins, energy_contributions_per_spin[idx_anisotropy].second);
		if (idx_bqc >= 0)        E_BQC(spins, energy_contributions_per_spin[idx_bqc].second);
		if (idx_fsc >= 0)        E_FourSC(spins, energy_contributions_per_spin[idx_fsc].second);
		if (idx_dmi >= 0)        E_DM(spins, energy_contributions_per_spin[idx_dmi].second);
		if (idx_dd >= 0)         E_DipoleDipole(spins, energy_contributions_per_spin[idx_dd].second);
	};

	void Hamiltonian_Isotropic::E_Zeeman(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			Energy[ispin] -= this->external_field_magnitude * this->external_field_normal.dot(spins[ispin]);
		}
	}//end Zeeman

	void Hamiltonian_Isotropic::E_Exchange(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			for (int shell = 0; shell < this->n_neigh_shells; ++shell)
			{
				for (int jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh)
				{
					int jspin = this->neigh[ispin][shell][jneigh];
					Energy[ispin] -= 0.5 * this->jij[shell] * spins[ispin].dot(spins[jspin]);
				}
			}
		}
	}//end Exchange

	void Hamiltonian_Isotropic::E_Anisotropic(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			Energy[ispin] -= this->anisotropy_magnitude * std::pow(this->anisotropy_normal.dot(spins[ispin]), 2.0);
		}
	}//end Anisotropic

	void Hamiltonian_Isotropic::E_BQC(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			int shell = 0;
			for (int jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh)
			{
				int jspin = this->neigh[ispin][shell][jneigh];
				Energy[ispin] -= 0.5 * this->bij * spins[ispin].dot(spins[jspin]);
			}
		}
	}//end BQC

	void Hamiltonian_Isotropic::E_FourSC(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			scalar result = 0.0;
			scalar products[6];
			for (int t = 0; t < this->n_4spin[ispin]; ++t)
			{
				int jspin = this->neigh_4spin[0][ispin][t];
				int kspin = this->neigh_4spin[1][ispin][t];
				int lspin = this->neigh_4spin[2][ispin][t];

				products[0] = spins[ispin].dot(spins[jspin]);
				products[1] = spins[kspin].dot(spins[lspin]);
				products[2] = spins[ispin].dot(spins[lspin]);
				products[3] = spins[jspin].dot(spins[kspin]);
				products[4] = spins[ispin].dot(spins[kspin]);
				products[5] = spins[jspin].dot(spins[lspin]);

				Energy[ispin] -= 0.25 * this->kijkl *
					(products[0] * products[1]
						+ products[2] * products[3]
						- products[4] * products[5]);
			}
		}
	}//end FourSC

	void Hamiltonian_Isotropic::E_DM(const vectorfield & spins, scalarfield & Energy)
	{
		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			int shell = 0;
			for (int jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh)
			{
				int jspin = this->neigh[ispin][shell][jneigh];
				Energy[ispin] -= 0.5 * this->dij * this->dm_normal[ispin][jneigh].dot(spins[ispin].cross(spins[jspin]));
			}
		}
	}// end DM

	void Hamiltonian_Isotropic::E_DipoleDipole(const vectorfield& spins, scalarfield & Energy)
	{
		scalar mult = -std::pow(Vectormath::MuB(),2) * 1.0 / 4.0 / M_PI * this->mu_s * this->mu_s; // multiply with mu_B^2

		for (unsigned int ispin = 0; ispin < spins.size(); ++ispin)
		{
			for (int jneigh = 0; jneigh < (int)this->dd_neigh[ispin].size(); ++jneigh)
			{
				if (dd_distance[ispin][jneigh] > 0.0)
				{
					int jspin = this->dd_neigh[ispin][jneigh];
					Energy[ispin] += 0.5 * mult / std::pow(dd_distance[ispin][jneigh], 3.0) *
						(3 * spins[jspin].dot(dd_normal[ispin][jneigh]) * spins[ispin].dot(dd_normal[ispin][jneigh]) - spins[ispin].dot(spins[jspin]));
				}
			}
		}
	}// end DipoleDipole

	void Hamiltonian_Isotropic::Gradient(const vectorfield & spins, vectorfield & gradient)
	{
		//========================= Init local vars ================================
		int nos = spins.size();
		int istart = -1, istop = istart + 1, i;
		if (istart == -1) { istart = 0; istop = nos; }
		std::vector<scalar> build_array = { 0.0, 0.0, 0.0 };
		std::vector<scalar> build_array_2 = { 0.0, 0.0, 0.0 };
		//Initialize field to { 0 }
		Vectormath::fill(gradient, {0,0,0});
		//------------------------ End Init ----------------------------------------
		if (this->dd_radius != 0.0)
		{
			for (i = istart; i < istop; ++i)
			{
				Field_Zeeman(nos, spins, gradient, i);
				Field_Exchange(nos, spins, gradient, i);
				Field_Anisotropic(nos, spins, gradient, i);
				Field_BQC(nos, spins, gradient, i);
				Field_FourSC(nos, spins, gradient, i);
				Field_DM(nos, spins, gradient, i);
				Field_DipoleDipole(nos, spins, gradient, i);
			}//endfor i
		}// endif dd_radius!=0.0
		else if (this->kijkl != 0.0 && this->dij != 0.0)
		{
			for (i = istart; i < istop; ++i)
			{
				Field_Zeeman(nos, spins, gradient, i);
				Field_Exchange(nos, spins, gradient, i);
				Field_Anisotropic(nos, spins, gradient, i);
				Field_BQC(nos, spins, gradient, i);
				Field_FourSC(nos, spins, gradient, i);
				Field_DM(nos, spins, gradient, i);
			}//endfor i
		}//endif kijkl != 0 & dij !=0
		else if (this->kijkl == 0.0 && this->dij == 0.0)
		{
			for (i = istart; i < istop; ++i)
			{
				Field_Zeeman(nos, spins, gradient, i);
				Field_Exchange(nos, spins, gradient, i);
				Field_Anisotropic(nos, spins, gradient, i);
				Field_BQC(nos, spins, gradient, i);
			}//endfor i
		}//endif kijkl == 0 & dij ==0
		else if (this->kijkl == 0.0 && this->dij != 0.0)
		{
			for (i = istart; i < istop; ++i)
			{
				Field_Zeeman(nos, spins, gradient, i);
				Field_Exchange(nos, spins, gradient, i);
				Field_Anisotropic(nos, spins, gradient, i);
				Field_BQC(nos, spins, gradient, i);
				Field_DM(nos, spins, gradient, i);
			}//endfor i
		}//endif kijkl == 0 & dij !=0
		else if (this->kijkl != 0.0 && this->dij == 0.0)
		{
			for (i = istart; i < istop; ++i)
			{
				Field_Zeeman(nos, spins, gradient, i);
				Field_Exchange(nos, spins, gradient, i);
				Field_Anisotropic(nos, spins, gradient, i);
				Field_BQC(nos, spins, gradient, i);
				Field_FourSC(nos, spins, gradient, i);
			}//endfor i
		}//endif kijkl != 0 & dij ==0
		// Turn the effective field into a gradient
		Vectormath::scale(gradient, -1);
	}

	void Hamiltonian_Isotropic::Field_Zeeman(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		eff_field[ispin] += this->external_field_magnitude*this->external_field_normal;
	}

	//Exchange Interaction
	void Hamiltonian_Isotropic::Field_Exchange(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		for (int shell = 0; shell < this->n_neigh_shells; ++shell)
		{
			for (int jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh)
			{
				int jspin = this->neigh[ispin][shell][jneigh];
				eff_field[ispin] += this->jij[shell] * spins[jspin];
			}
		}
	}//end Exchange

	 //Anisotropy
	void Hamiltonian_Isotropic::Field_Anisotropic(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		eff_field[ispin] += 2 * this->anisotropy_magnitude*this->anisotropy_normal * this->anisotropy_normal.dot(spins[ispin]);
	}//end Anisotropic

	 // Biquadratic Coupling
	void Hamiltonian_Isotropic::Field_BQC(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		int shell = 0;
		for (int jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh)
		{
			int jspin = this->neigh[ispin][shell][jneigh];
			eff_field[ispin] += 2 * this->bij * spins[jspin] * spins[ispin].dot(spins[jspin]);
		}
	}//end BQC

	 // Four Spin Interaction
	void Hamiltonian_Isotropic::Field_FourSC(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		for (int t = 0; t < this->n_4spin[ispin]; ++t)
		{
			int jspin = this->neigh_4spin[0][ispin][t];
			int kspin = this->neigh_4spin[0][ispin][t];
			int lspin = this->neigh_4spin[0][ispin][t];
			eff_field[ispin] += this->kijkl
				* spins[jspin] * spins[kspin].dot(spins[lspin])
				+ spins[lspin] * spins[jspin].dot(spins[kspin])
				- spins[kspin] * spins[jspin].dot(spins[lspin]);
		}
	}//end FourSC effective field

	 // Dzyaloshinskii-Moriya Interaction 
	void Hamiltonian_Isotropic::Field_DM(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		int shell = 0;
		for (int jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh)
		{
			int jspin = this->neigh[ispin][shell][jneigh];
			eff_field[ispin] += this->dij * this->dm_normal[ispin][jneigh].cross(spins[jspin]);
		}
	}//end DM effective Field

	void Hamiltonian_Isotropic::Field_DipoleDipole(int nos, const vectorfield & spins, vectorfield & eff_field, const int ispin)
	{
		scalar mult = 1.0 / 4.0 / M_PI * this->mu_s * this->mu_s; // multiply with mu_B^2
		for (int jneigh = 0; jneigh < (int)this->dd_neigh[ispin].size(); ++jneigh)
		{
			if (dd_distance[ispin][jneigh] > 0.0)
			{
				int jspin = this->dd_neigh[ispin][jneigh];
				scalar skalar_contrib = mult / std::pow(dd_distance[ispin][jneigh], 3.0);
				eff_field[ispin] += skalar_contrib * (3 * dd_normal[ispin][jneigh]*spins[jspin].dot(dd_normal[ispin][jneigh]) - spins[jspin]);
			}
		}
	}//end Field_DipoleDipole

	void Hamiltonian_Isotropic::Hessian(const vectorfield & spins, MatrixX & hessian)
	{
		//int nos = spins.size() / 3;

		//// Single Spin elements
		//for (int alpha = 0; alpha < 3; ++alpha)
		//{
		//	scalar K = 2.0*this->anisotropy_magnitude*this->anisotropy_normal[alpha];
		//	for (int i = 0; i < nos; ++i)
		//	{
		//		hessian[i + alpha*nos + 3 * nos*(i + alpha*nos)] = K;
		//	}
		//}
	}

	// Hamiltonian name as string
	static const std::string name = "Isotropic Heisenberg";
	const std::string& Hamiltonian_Isotropic::Name() { return name; }
}