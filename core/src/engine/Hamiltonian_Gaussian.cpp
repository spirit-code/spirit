#include <engine/Hamiltonian_Gaussian.h>

#include <utility/Manifoldmath.h>
#include <utility/Vectormath.h>

using namespace Data;

namespace Engine
{
	Hamiltonian_Gaussian::Hamiltonian_Gaussian(
		std::vector<double> amplitude, std::vector<double> width, std::vector<std::vector<double>> center
	) :
		Hamiltonian(std::vector<bool>{false, false, false}), amplitude(amplitude), width(width), center(center)
	{
		this->n_gaussians = amplitude.size();
	}


	void Hamiltonian_Gaussian::Hessian(const std::vector<double> & spins, std::vector<double> & hessian)
	{
		int nos = spins.size() / 3;
		for (int ispin = 0; ispin < nos; ++ispin)
		{
			// Set Hessian to zero
			for (int alpha = 0; alpha < 3; ++alpha)
			{
				for (int beta = 0; beta < 3; ++beta)
				{
					int idx_h = ispin + alpha*nos + 3 * nos*(ispin + beta*nos);
					hessian[idx_h] = 0.0;
				}
			}
			// Calculate Hessian
			for (int i = 0; i < this->n_gaussians; ++i)
			{
				// Distance between spin and gaussian center
				std::vector<double> n{ spins[ispin], spins[ispin + nos], spins[ispin + 2 * nos] };
				double l = 1.0 - Utility::Vectormath::Dot_Product(this->center[i], n); //Utility::Manifoldmath::Dist_Greatcircle(this->center[i], n);
				// Scalar product of spin and gaussian center
				//double nc = 0;
				//for (int dim = 0; dim < 3; ++dim) nc += spins[ispin + dim*nos] * this->center[i][dim];
				// Prefactor for all alpha, beta
				double prefactor = this->amplitude[i] * std::exp(-std::pow(l, 2) / (2.0*std::pow(this->width[i], 2)))
					/ std::pow(this->width[i], 2)
					* (std::pow(l, 2) / std::pow(this->width[i], 2) - 1);
				// Effective Field contribution
				for (int alpha = 0; alpha < 3; ++alpha)
				{
					for (int beta = 0; beta < 3; ++beta)
					{
						int idx_h = ispin + alpha*nos + 3 * nos*(ispin + beta*nos);
						hessian[idx_h] += prefactor * this->center[i][alpha] * this->center[i][beta];
					}
				}
			}
		}
	}

	void Hamiltonian_Gaussian::Effective_Field(const std::vector<double> & spins, std::vector<double> & field)
	{
		int nos = spins.size() / 3;

		for (int ispin = 0; ispin < nos; ++ispin)
		{
			// Set field to zero
			field[ispin] = 0; field[ispin + nos] = 0; field[ispin + 2 * nos] = 0;
			// Calculate field
			for (int i = 0; i < this->n_gaussians; ++i)
			{
				// Distance between spin and gaussian center
				std::vector<double> n { spins[ispin], spins[ispin + nos], spins[ispin + 2 * nos] };
				double l = 1.0 - Utility::Vectormath::Dot_Product(this->center[i], n); //Utility::Manifoldmath::Dist_Greatcircle(this->center[i], n);
				// Scalar product of spin and gaussian center
				//double nc = 0;
				//for (int dim = 0; dim < 3; ++dim) nc += spins[ispin + dim*nos] * this->center[i][dim];
				// Prefactor
				double prefactor = this->amplitude[i] * std::exp(-std::pow(l, 2) / (2.0*std::pow(this->width[i], 2))) * l / std::pow(this->width[i],2);
				// Effective Field contribution
				for (int dim = 0; dim < 3; ++dim)
				{
					field[ispin + dim*nos] -= prefactor * this->center[i][dim];
				}
			}
		}
	}

	double Hamiltonian_Gaussian::Energy(const std::vector<double> & spins)
	{
		int nos = spins.size() / 3;
		double E = 0;

		for (int i = 0; i < this->n_gaussians; ++i)
		{
			for (int ispin = 0; ispin < nos; ++ispin)
			{
				// Distance between spin and gaussian center
				std::vector<double> n{ spins[ispin], spins[ispin + nos], spins[ispin + 2 * nos] };
				double l = 1.0 - Utility::Vectormath::Dot_Product(this->center[i], n); //Utility::Manifoldmath::Dist_Greatcircle(this->center[i], n);
				// Energy contribution
				E += this->amplitude[i] * std::exp( -std::pow(l, 2)/(2.0*std::pow(this->width[i], 2)) );
			}
		}

		return E;
	}

	std::vector<double> Hamiltonian_Gaussian::Energy_Array(const std::vector<double> & spins)
	{
		return std::vector<double>(9, 0.0);
	}

	//std::vector<std::vector<double>> Energy_Array_per_Spin(std::vector<double> & spins) override;

	// Hamiltonian name as string
	std::string Hamiltonian_Gaussian::Name() { return "Gaussian"; }
}