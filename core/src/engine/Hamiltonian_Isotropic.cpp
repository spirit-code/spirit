#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include "Hamiltonian_Isotropic.h"
#include "Vectormath.h"
#include "Vectoroperators.h"
#include "Neighbours.h"
#include "Logging.h"
//extern Utility::LoggingHandler Log;

using namespace Utility;

namespace Engine
{
	Hamiltonian_Isotropic::Hamiltonian_Isotropic(
		std::vector<bool> boundary_conditions, double external_field_magnitude_i, std::vector<double> external_field_normal, double mu_s,
		double anisotropy_magnitude, std::vector<double> anisotropy_normal,
		int n_neigh_shells, std::vector<double> jij, double dij, double bij, double kijkl, double dd_radius,
		Data::Geometry geometry) :
		Hamiltonian(boundary_conditions),
		mu_s(mu_s),
		external_field_magnitude(external_field_magnitude_i), external_field_normal(external_field_normal),
		anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
		n_neigh_shells(n_neigh_shells), jij(jij), dij(dij), bij(bij), kijkl(kijkl), dd_radius(dd_radius)
	{
		// Rescale magnetic field from Tesla to meV
		external_field_magnitude = external_field_magnitude * Utility::Vectormath::MuB() * mu_s;
		Utility::Vectormath::Normalize(external_field_normal);

		// Calculate Neighbours
		Log(Log_Level::Info, Log_Sender::All, "Building Neighbours ...");
		Engine::Neighbours::Create_Neighbours(geometry, boundary_conditions, n_neigh_shells,
			n_spins_in_shell, neigh, n_4spin, max_n_4spin, neigh_4spin, dm_normal, segments, segments_pos);
		Engine::Neighbours::Create_Dipole_Neighbours(geometry, boundary_conditions,
			dd_radius, dd_neigh, dd_neigh_pos, dd_normal, dd_distance);
		Log(Log_Level::Info, Log_Sender::All, "Done Caclulating Neighbours");
	}


	double Hamiltonian_Isotropic::Energy(std::vector<double> & spins)
	{
		return sum(Energy_Array(spins));
	}

	std::vector<double> Hamiltonian_Isotropic::Energy_Array(std::vector<double> & spins)
	{
		//========================= Init local vars ================================
		int nos = spins.size() / 3;
		int i = 0, istart = -1, istop = istart + 1;
		std::vector<double> energies(7, 0.0);
		double f0 = 1.0;
		if (istart == -1) { istart = 0; istop = nos; f0 = 0.5; }
		//------------------------ End Init ----------------------------------------
		if (this->dd_radius != 0.0) {
			for (i = istart; i < istop; ++i) {
				energies[ENERGY_POS_ZEEMAN] = energies[ENERGY_POS_ZEEMAN] + E_Zeeman(nos, spins, i);
				energies[ENERGY_POS_EXCHANGE] = energies[ENERGY_POS_EXCHANGE] + E_Exchange(nos, spins, i) *f0;
				energies[ENERGY_POS_ANISOTROPY] = energies[ENERGY_POS_ANISOTROPY] + E_Anisotropic(nos, spins, i);
				energies[ENERGY_POS_BQC] = energies[ENERGY_POS_BQC] + E_BQC(nos, spins, i)*f0;
				energies[ENERGY_POS_FSC] = energies[ENERGY_POS_FSC] + E_FourSC(nos, spins, i)*f0*f0;
				energies[ENERGY_POS_DMI] = energies[ENERGY_POS_DMI] + E_DM(nos, spins, i)*f0;
				energies[ENERGY_POS_DD] = energies[ENERGY_POS_DD] + E_DipoleDipole(nos, spins, i);
			}//endfor i
		}
		else if (this->kijkl != 0.0 && this->dij != 0.0) {
			for (i = istart; i < istop; ++i) {
				energies[ENERGY_POS_ZEEMAN] = energies[ENERGY_POS_ZEEMAN] + E_Zeeman(nos, spins, i);
				energies[ENERGY_POS_EXCHANGE] = energies[ENERGY_POS_EXCHANGE] + E_Exchange(nos, spins, i) *f0;
				energies[ENERGY_POS_ANISOTROPY] = energies[ENERGY_POS_ANISOTROPY] + E_Anisotropic(nos, spins, i);
				energies[ENERGY_POS_BQC] = energies[ENERGY_POS_BQC] + E_BQC(nos, spins, i)*f0;
				energies[ENERGY_POS_FSC] = energies[ENERGY_POS_FSC] + E_FourSC(nos, spins, i)*f0*f0;
				energies[ENERGY_POS_DMI] = energies[ENERGY_POS_DMI] + E_DM(nos, spins, i)*f0;
			}//endfor i
		}//endif kijkl != 0 & dij !=0
		else if (this->kijkl == 0.0 && this->dij == 0.0) {
			for (i = istart; i < istop; ++i) {
				energies[ENERGY_POS_ZEEMAN] = energies[ENERGY_POS_ZEEMAN] + E_Zeeman(nos, spins, i);
				energies[ENERGY_POS_EXCHANGE] = energies[ENERGY_POS_EXCHANGE] + E_Exchange(nos, spins, i) *f0;
				energies[ENERGY_POS_ANISOTROPY] = energies[ENERGY_POS_ANISOTROPY] + E_Anisotropic(nos, spins, i);
				energies[ENERGY_POS_BQC] = energies[ENERGY_POS_BQC] + E_BQC(nos, spins, i)*f0;
			}//endfor i
		}//endif kijkl == 0 & dij ==0
		else if (this->kijkl == 0.0 && this->dij != 0.0) {
			for (i = istart; i < istop; ++i) {
				energies[ENERGY_POS_ZEEMAN] = energies[ENERGY_POS_ZEEMAN] + E_Zeeman(nos, spins, i);
				energies[ENERGY_POS_EXCHANGE] = energies[ENERGY_POS_EXCHANGE] + E_Exchange(nos, spins, i) *f0;
				energies[ENERGY_POS_ANISOTROPY] = energies[ENERGY_POS_ANISOTROPY] + E_Anisotropic(nos, spins, i);
				energies[ENERGY_POS_BQC] = energies[ENERGY_POS_BQC] + E_BQC(nos, spins, i)*f0;
				energies[ENERGY_POS_DMI] = energies[ENERGY_POS_DMI] + E_DM(nos, spins, i)*f0;
			}//endfor i
		}//endif kijkl == 0 & dij !=0
		else if (this->kijkl != 0.0 && this->dij == 0.0) {
			for (i = istart; i < istop; ++i) {
				energies[ENERGY_POS_ZEEMAN] = energies[ENERGY_POS_ZEEMAN] + E_Zeeman(nos, spins, i);
				energies[ENERGY_POS_EXCHANGE] = energies[ENERGY_POS_EXCHANGE] + E_Exchange(nos, spins, i) *f0;
				energies[ENERGY_POS_ANISOTROPY] = energies[ENERGY_POS_ANISOTROPY] + E_Anisotropic(nos, spins, i);
				energies[ENERGY_POS_BQC] = energies[ENERGY_POS_BQC] + E_BQC(nos, spins, i)*f0;
				energies[ENERGY_POS_FSC] = energies[ENERGY_POS_FSC] + E_FourSC(nos, spins, i)*f0*f0;
			}//endfor i
		}//endif kijkl != 0 & Utility::Vectormath::Length(s->dij, s->n_shells) ==0
		return energies;
	};//end Total_Array_

	double Hamiltonian_Isotropic::E_Zeeman(int nos, std::vector<double> & spins, const int ispin)
	{
		double dotProduct = 0.0;
		for (int dim = 0; dim < 3; ++dim) {
			dotProduct += spins[dim*nos + ispin] * this->external_field_normal[dim];
		}
		return -dotProduct*this->external_field_magnitude;
	}//end Zeeman

	double Hamiltonian_Isotropic::E_Exchange(int nos, std::vector<double> & spins, const int ispin)
	{
		//========================= Init local vars ================================
		double result = 0.0, dotProduct;
		int shell, jneigh, jspin, dim;
		//------------------------ End Init ----------------------------------------
		for (shell = 0; shell < this->n_neigh_shells; ++shell) {
			for (jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh) {
				jspin = this->neigh[ispin][shell][jneigh];
				dotProduct = 0.0;
				for (dim = 0; dim < 3; ++dim) { dotProduct += spins[dim*nos + ispin] * spins[dim*nos + jspin]; }
				result -= this->jij[shell] * dotProduct;
			}
		}
		return result;
	}//end Exchange

	double Hamiltonian_Isotropic::E_Anisotropic(int nos, std::vector<double> & spins, const int ispin)
	{
		double dotProduct = 0.0;
		for (int dim = 0; dim < 3; ++dim) { dotProduct += this->anisotropy_normal[dim] * spins[dim*nos + ispin]; }
		return -this->anisotropy_magnitude * std::pow(dotProduct, 2.0);
	}//end Anisotropic

	double Hamiltonian_Isotropic::E_BQC(int nos, std::vector<double> & spins, const int ispin)
	{
		//========================= Init local vars ================================
		double result = 0.0, dotProduct;
		int shell = 0, jneigh, jspin, dim;
		//------------------------ End Init ----------------------------------------
		for (jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh) {
			jspin = this->neigh[ispin][shell][jneigh];
			dotProduct = 0.0;
			for (dim = 0; dim < 3; ++dim) { dotProduct += spins[dim*nos + ispin] * spins[dim*nos + jspin]; }
			result = result - this->bij * dotProduct;
		}
		return result;
	}//end BQC

	double Hamiltonian_Isotropic::E_FourSC(int nos, std::vector<double> & spins, const int ispin)
	{
		//========================= Init local vars ================================
		double result = 0.0;
		std::vector<double> dotProduct = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		int t, jspin, kspin, lspin, dim;
		//------------------------ End Init ----------------------------------------
		for (t = 0; t < this->n_4spin[ispin]; ++t) {
			jspin = this->neigh_4spin[0][ispin][t];
			kspin = this->neigh_4spin[1][ispin][t];
			lspin = this->neigh_4spin[2][ispin][t];
			dotProduct = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			for (dim = 0; dim < 3; ++dim) {
				dotProduct[0] += spins[dim*nos + ispin] * spins[dim*nos + jspin];
				dotProduct[1] += spins[dim*nos + kspin] * spins[dim*nos + lspin];
				dotProduct[2] += spins[dim*nos + ispin] * spins[dim*nos + lspin];
				dotProduct[3] += spins[dim*nos + jspin] * spins[dim*nos + kspin];
				dotProduct[4] += spins[dim*nos + ispin] * spins[dim*nos + kspin];
				dotProduct[5] += spins[dim*nos + jspin] * spins[dim*nos + lspin];
			}
			result = result - this->kijkl *
				(dotProduct[0] * dotProduct[1]
					+ dotProduct[2] * dotProduct[3]
					- dotProduct[4] * dotProduct[5]);
		}
		return result;
	}//end FourSC

	double Hamiltonian_Isotropic::E_DM(int nos, std::vector<double> & spins, const int ispin)
	{
		//========================= Init local vars ================================
		double result = 0.0, crossProduct, dotProduct;
		int shell = 0, jneigh, jspin, dim;
		double build_array[3] = { 0 };
		//------------------------ End Init ----------------------------------------
		for (jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh) {
			jspin = this->neigh[ispin][shell][jneigh];
			dotProduct = 0.0;
			for (dim = 0; dim < 3; ++dim) {
				crossProduct = spins[((dim + 1) % 3)*nos + ispin] * spins[((dim + 2) % 3)*nos + jspin]
					- spins[((dim + 2) % 3)*nos + ispin] * spins[((dim + 1) % 3)*nos + jspin];
				dotProduct += this->dm_normal[dim][ispin][jneigh] * crossProduct;
			}
			result = result - this->dij * dotProduct;
		}
		return result;
	}// end DM
	double Hamiltonian_Isotropic::E_DipoleDipole(int nos, std::vector<double>& spins, const int ispin)
	{
		//========================= Init local vars ================================
		int jneigh, jspin;
		double mult = -Utility::Vectormath::MuB()*Utility::Vectormath::MuB()*1.0 / 4.0 / M_PI * this->mu_s * this->mu_s; // multiply with mu_B^2
		double result = 0.0;
		//------------------------ End Init ----------------------------------------
		for (jneigh = 0; jneigh < (int)this->dd_neigh[ispin].size(); ++jneigh) {
			if (dd_distance[ispin][jneigh] > 0.0) {
				jspin = this->dd_neigh[ispin][jneigh];
				result += mult / std::pow(dd_distance[ispin][jneigh], 3.0) *
					(3 * (spins[jspin] * dd_normal[0][ispin][jneigh]
						+ spins[1 * nos + jspin] * dd_normal[1][ispin][jneigh]
						+ spins[2 * nos + jspin] * dd_normal[2][ispin][jneigh])
						*
						(spins[ispin] * dd_normal[0][ispin][jneigh]
							+ spins[1 * nos + ispin] * dd_normal[1][ispin][jneigh]
							+ spins[2 * nos + ispin] * dd_normal[2][ispin][jneigh])
						-
						(spins[ispin] * spins[jspin]
							+ spins[1 * nos + ispin] * spins[1 * nos + jspin]
							+ spins[2 * nos + ispin] * spins[2 * nos + jspin]));
			}
		}//endfor jneigh
		return result;
	}// end DipoleDipole

	void Hamiltonian_Isotropic::Effective_Field(const std::vector<double> & spins, std::vector<double> & field)
	{
		//========================= Init local vars ================================
		int nos = spins.size()/3;
		int dim, istart = -1, istop = istart + 1, i;
		if (istart == -1) { istart = 0; istop = nos; }
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };
		std::vector<double> build_array_2 = { 0.0, 0.0, 0.0 };
		//Initialize field to { 0 }
		for (i = 0; i < nos; ++i) {
			field[i] = 0; field[nos + i] = 0; field[2 * nos + i] = 0;
		}
		//------------------------ End Init ----------------------------------------
		if (this->dd_radius != 0.0) {
			for (i = istart; i < istop; ++i) {
				for (dim = 0; dim < 3; ++dim) { build_array[dim] = field[dim*nos + i]; }
				Field_Zeeman(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Exchange(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Anisotropic(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_BQC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_FourSC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_DM(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_DipoleDipole(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				for (dim = 0; dim < 3; ++dim) { field[dim*nos + i] = build_array[dim]; }
			}//endfor i
		}// endif dd_radius!=0.0
		else if (this->kijkl != 0.0 && this->dij != 0.0) {
			for (i = istart; i < istop; ++i) {
				for (dim = 0; dim < 3; ++dim) { build_array[dim] = field[dim*nos + i]; }
				Field_Zeeman(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Exchange(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Anisotropic(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_BQC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_FourSC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_DM(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				for (dim = 0; dim < 3; ++dim) { field[dim*nos + i] = build_array[dim]; }
			}//endfor i
		}//endif kijkl != 0 & dij !=0
		else if (this->kijkl == 0.0 && this->dij == 0.0) {
			for (i = istart; i < istop; ++i) {
				for (dim = 0; dim < 3; ++dim) { build_array[dim] = field[dim*nos + i]; }
				Field_Zeeman(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Exchange(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Anisotropic(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_BQC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				for (dim = 0; dim < 3; ++dim) { field[dim*nos + i] = build_array[dim]; }
			}//endfor i
		}//endif kijkl == 0 & dij ==0
		else if (this->kijkl == 0.0 && this->dij != 0.0) {
			for (i = istart; i < istop; ++i) {
				for (dim = 0; dim < 3; ++dim) { build_array[dim] = field[dim*nos + i]; }
				Field_Zeeman(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Exchange(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Anisotropic(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_BQC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_DM(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				for (dim = 0; dim < 3; ++dim) { field[dim*nos + i] = build_array[dim]; }
			}//endfor i
		}//endif kijkl == 0 & dij !=0
		else if (this->kijkl != 0.0 && this->dij == 0.0) {
			for (i = istart; i < istop; ++i) {
				for (dim = 0; dim < 3; ++dim) { build_array[dim] = field[dim*nos + i]; }
				Field_Zeeman(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Exchange(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_Anisotropic(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_BQC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				Field_FourSC(nos, spins, build_array_2, i);
				Vectormath::Array_Array_Add(build_array, build_array_2, 1.0);
				for (dim = 0; dim < 3; ++dim) { field[dim*nos + i] = build_array[dim]; }
			}//endfor i
		}//endif kijkl != 0 & dij ==0
	}

	void Hamiltonian_Isotropic::Field_Zeeman(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		for (int dim = 0; dim < 3; ++dim) {
			eff_field[dim] = this->external_field_normal[dim] * this->external_field_magnitude;
		}
	}

	//Exchange Interaction
	void Hamiltonian_Isotropic::Field_Exchange(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		//========================= Init local vars ================================
		int dim, shell, jneigh, jspin;
		for (dim = 0; dim < 3; ++dim) { eff_field[dim] = 0; }
		//------------------------ End Init ----------------------------------------
		for (shell = 0; shell < this->n_neigh_shells; ++shell) {
			for (jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh) {
				jspin = this->neigh[ispin][shell][jneigh];
				for (dim = 0; dim < 3; ++dim) {
					eff_field[dim] += spins[dim*nos + jspin] * this->jij[shell];
				}
			}//endfor jneigh
		}//endfor shell
	}//end Exchange

	 //Anisotropy
	void Hamiltonian_Isotropic::Field_Anisotropic(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		//========================= Init local vars ================================
		int dim;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };
		for (dim = 0; dim < 3; ++dim) { build_array[dim] = spins[dim*nos + ispin]; }
		double mult = 2.0 * this->anisotropy_magnitude * Vectormath::Dot_Product(this->anisotropy_normal, build_array);
		//------------------------ End Init ----------------------------------------
		for (dim = 0; dim < 3; ++dim) {
			eff_field[dim] = this->anisotropy_normal[dim] * mult;
		}
	}//end Anisotropic

	 // Biquadratic Coupling
	void Hamiltonian_Isotropic::Field_BQC(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		//========================= Init local vars ================================
		int shell = 0, jneigh, jspin, dim;
		double dotProduct = 0.0;
		for (dim = 0; dim < 3; ++dim) { eff_field[dim] = 0; }
		//------------------------ End Init ----------------------------------------
		for (jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh) {
			jspin = this->neigh[ispin][shell][jneigh];
			dotProduct = 0.0;
			for (dim = 0; dim < 3; ++dim) {
				dotProduct += spins[dim*nos + ispin] * spins[dim*nos + jspin];
			}
			for (dim = 0; dim < 3; ++dim) {
				eff_field[dim] += spins[dim*nos + jspin] * 2.0 * this->bij * dotProduct;
			}
		}//enfor jneigh
	}//end BQC

	 // Four Spin Interaction
	void Hamiltonian_Isotropic::Field_FourSC(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		//========================= Init local vars ================================
		int dim, jspin, kspin, lspin, t;
		double dp_1, dp_2, dp_3;
		std::vector<double> build_array = { 0.0, 0.0, 0.0 };
		std::vector<double> build_array_2 = { 0.0, 0.0, 0.0 };
		for (dim = 0; dim < 3; ++dim) { eff_field[dim] = 0; }
		//------------------------ End Init ----------------------------------------
		for (t = 0; t < this->n_4spin[ispin]; ++t) {
			jspin = this->neigh_4spin[0][ispin][t];
			kspin = this->neigh_4spin[0][ispin][t];
			lspin = this->neigh_4spin[0][ispin][t];
			/*
			eff_field(:) = eff_field(:) &
			+Kijkl*( SPINS(jspin,:)*DOT_PRODUCT(SPINS(kspin,:),SPINS(lspin,:)) &
			+SPINS(lspin,:)*DOT_PRODUCT(SPINS(jspin,:),SPINS(kspin,:)) &
			-SPINS(kspin,:)*DOT_PRODUCT(SPINS(jspin,:),SPINS(lspin,:)) )
			*/
			dp_1 = 0.0; dp_2 = 0.0; dp_3 = 0.0;
			for (dim = 0; dim < 3; ++dim) {
				dp_1 += spins[dim*nos + kspin] * spins[dim*nos + lspin];
				dp_2 += spins[dim*nos + jspin] * spins[dim*nos + kspin];
				dp_3 += spins[dim*nos + jspin] * spins[dim*nos + lspin];
			}
			for (dim = 0; dim < 3; ++dim)
			{
				eff_field[dim] += this->kijkl * (
					spins[dim*nos + jspin] * dp_1
					+ spins[dim*nos + lspin] * dp_2
					- spins[dim*nos + kspin] * dp_3
					);
			}
		}//endfor t
	}//end FourSC effective field

	 // Dzyaloshinskii-Moriya Interaction 
	void Hamiltonian_Isotropic::Field_DM(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		//========================= Init local vars ================================
		int dim, shell = 0, jneigh, jspin;
		double build_array[3] = { 0 };
		for (dim = 0; dim < 3; ++dim) { eff_field[dim] = 0; }
		//------------------------ End Init ----------------------------------------
		for (jneigh = 0; jneigh < this->n_spins_in_shell[ispin][shell]; ++jneigh) {
			jspin = this->neigh[ispin][shell][jneigh];
			for (dim = 0; dim < 3; ++dim) {
				eff_field[dim] += this->dij *
					(spins[((dim + 1) % 3)*nos + jspin] * this->dm_normal[((dim + 2) % 3)][ispin][jneigh]
						- spins[((dim + 2) % 3)*nos + jspin] * this->dm_normal[((dim + 1) % 3)][ispin][jneigh]);
			}
		}//endfor jneigh
	}//end DM effective Field

	void Hamiltonian_Isotropic::Field_DipoleDipole(int nos, const std::vector<double> & spins, std::vector<double> & eff_field, const int ispin)
	{
		//========================= Init local vars ================================
		eff_field[0] = 0.0; eff_field[1] = 0.0; eff_field[2] = 0.0;
		int dim, jneigh, jspin;
		double mult = 1.0 / 4.0 / M_PI * this->mu_s * this->mu_s; // multiply with mu_B^2
		double skalar_contrib, dotprod;
		//------------------------ End Init ----------------------------------------
		for (jneigh = 0; jneigh < (int)this->dd_neigh[ispin].size(); ++jneigh) {
			if (dd_distance[ispin][jneigh] > 0.0) {
				jspin = this->dd_neigh[ispin][jneigh];
				skalar_contrib = mult / std::pow(dd_distance[ispin][jneigh], 3.0);
				dotprod = spins[jspin] * dd_normal[0][ispin][jneigh]
					+ spins[1 * nos + jspin] * dd_normal[1][ispin][jneigh]
					+ spins[2 * nos + jspin] * dd_normal[2][ispin][jneigh];
				for (dim = 0; dim < 3; ++dim) {
					eff_field[dim] += skalar_contrib * (3 * dotprod*dd_normal[dim][ispin][jneigh] - spins[dim * nos + jspin]);
				}
			}
		}//endfor jneigh
	}//end Field_DipoleDipole
}