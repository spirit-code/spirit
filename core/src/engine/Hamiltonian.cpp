#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

namespace Engine
{
	Hamiltonian::Hamiltonian(std::vector<bool> boundary_conditions) :
        boundary_conditions(boundary_conditions)
    {
        prng = std::mt19937(94199188);
		distribution_int = std::uniform_int_distribution<int>(0, 1);

        delta = 1e-6;
    }


	void Hamiltonian::Update_Energy_Contributions()
	{
		// Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Update_Energy_Contributions() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
	}


    // void Hamiltonian::Hessian(const std::vector<scalar> & spins, std::vector<scalar> & hessian)
    // {
	// 	// This is a regular finite difference implementation (probably not very efficient)
	//  // using the differences between function values (not gradient)
	// 	// see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

	// 	int nos = spins.size() / 3;

	// 	// Calculate finite difference
	// 	std::vector<scalar> spins_pp(3 * nos, 0);
	// 	std::vector<scalar> spins_mm(3 * nos, 0);
	// 	std::vector<scalar> spins_pm(3 * nos, 0);
	// 	std::vector<scalar> spins_mp(3 * nos, 0);

	// 	for (int i = 0; i < 3 * nos; ++i)
	// 	{
	// 		for (int j = 0; j < 3 * nos; ++j)
	// 		{
	// 			if (i == j)
	// 			{
	// 				spins_pp = spins;
	// 				spins_mm = spins;
	// 				spins_pm = spins;
	// 				spins_mp = spins;

	// 				spins_pp[i] = spins_pp[i] + 2.0*delta;
	// 				spins_mm[i] = spins_mm[i] - 2.0*delta;
	// 				spins_pm[i] = spins_mm[i] + delta;
	// 				spins_mp[i] = spins_mm[i] - delta;

	// 				Utility::Vectormath::Normalize_3Nos(spins_pp);
	// 				Utility::Vectormath::Normalize_3Nos(spins_mm);
	// 				Utility::Vectormath::Normalize_3Nos(spins_pm);
	// 				Utility::Vectormath::Normalize_3Nos(spins_mp);

	// 				scalar E_pp = this->Energy(spins_pp);
	// 				scalar E_mm = this->Energy(spins_mm);
	// 				scalar E_pm = this->Energy(spins_pm);
	// 				scalar E_mp = this->Energy(spins_mp);
	// 				scalar E = this->Energy(spins);

	// 				hessian[i * 3 * nos + j] = (-E_pp +16*E_pm - 30*E + 16*E_mp - E_mm) / (12 * delta*delta);
	// 			}
	// 			else
	// 			{
	// 				spins_pp = spins;
	// 				spins_mm = spins;
	// 				spins_pm = spins;
	// 				spins_mp = spins;

	// 				spins_pp[i] = spins_pp[i] + delta;
	// 				spins_pp[j] = spins_pp[j] + delta;
	// 				spins_mm[i] = spins_mm[i] - delta;
	// 				spins_mm[j] = spins_mm[j] - delta;
	// 				spins_pm[i] = spins_mm[i] + delta;
	// 				spins_pm[j] = spins_mm[j] - delta;
	// 				spins_mp[i] = spins_mm[i] - delta;
	// 				spins_mp[j] = spins_mm[j] + delta;

	// 				Utility::Vectormath::Normalize_3Nos(spins_pp);
	// 				Utility::Vectormath::Normalize_3Nos(spins_mm);
	// 				Utility::Vectormath::Normalize_3Nos(spins_pm);
	// 				Utility::Vectormath::Normalize_3Nos(spins_mp);

	// 				scalar E_pp = this->Energy(spins_pp);
	// 				scalar E_mm = this->Energy(spins_mm);
	// 				scalar E_pm = this->Energy(spins_pm);
	// 				scalar E_mp = this->Energy(spins_mp);

	// 				hessian[i * 3 * nos + j] = (E_pp - E_pm - E_mp + E_mm) / (4 * delta*delta);
	// 			}
	// 		}
	// 	}
    // }

	void Hamiltonian::Hessian(const vectorfield & spins, MatrixX & hessian)
	{
		// This is a regular finite difference implementation (probably not very efficient)
		// using the differences between gradient values (not function)
		// see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

		int nos = spins.size();

		// Calculate finite difference
		vectorfield spins_p(nos);
		vectorfield spins_m(nos);

		std::vector<vectorfield> grad_p(3*nos, vectorfield(nos));
		std::vector<vectorfield> grad_m(3*nos, vectorfield(nos));

		scalarfield d(3 * nos);

		for (int i = 0; i < nos; ++i)
		{
			for (int dim = 0; dim < 3; ++dim)
			{
				spins_p = spins;
				spins_p[i][dim] += delta;
				spins_p[i].normalize();
				//Vectormath::Normalize_3Nos(spins_p);

				spins_m = spins;
				spins_m[i][dim] -= delta;
				spins_m[i].normalize();
				//Vectormath::Normalize_3Nos(spins_m);

				d[i + dim*nos] = Manifoldmath::dist_greatcircle(spins_m[i], spins_p[i]);
				//d[i + dim*nos] = Utility::Manifoldmath::Dist_Geodesic(spins_m, spins_p);
				if (d[i + dim*nos] > 0)
				{
					this->Gradient(spins_p, grad_p[i + dim*nos]);
					this->Gradient(spins_m, grad_m[i + dim*nos]);
				}
				else d[i + dim*nos] = 1;
			}
		}

		for (int i = 0; i < 3 * nos; ++i)
		{
			for (int dimi = 0; dimi < 3 * nos; ++dimi)
			{
				for (int j = 0; j < 3 * nos; ++j)
				{
					for (int dimj = 0; dimj < 3 * nos; ++dimj)
					{
						hessian(i + dimi*nos, j + dimj*nos) = (grad_p[i+dimi*nos][j][dimj] - grad_m[i+dimi*nos][j][dimj]) / (2 * d[i+dimi*nos]) + (grad_p[j + dimj*nos][i][dimi] - grad_m[j + dimj*nos][i][dimi]) / (2 * d[j+dimj*nos]);
					}
				}
			}
		}
	}

    void Hamiltonian::Gradient(const vectorfield & spins, vectorfield & gradient)
    {
		// This is a regular finite difference implementation (probably not very efficient)

        int nos = spins.size();

		// Calculate finite difference
		vectorfield spins_plus(nos);
		vectorfield spins_minus(nos);

		for (int i = 0; i < nos; ++i)
		{
			for (int dim = 0; dim < 3; ++dim)
			{
				spins_plus = spins;
				spins_minus = spins;

				spins_plus[i][dim] += delta;
				spins_minus[i][dim] -= delta;

				spins_plus[i].normalize();
				spins_minus[i].normalize();

				scalar d = Manifoldmath::dist_greatcircle(spins_minus[i], spins_plus[i]);

				if (d > 0)
				{
					scalar E_plus = this->Energy(spins_plus);
					scalar E_minus = this->Energy(spins_minus);
					gradient[i][dim] = (E_plus - E_minus) / d;
				}
				else gradient[i][dim] = 0;
			}
		}
    }

	scalar Hamiltonian::Energy(const vectorfield & spins)
	{
		scalar sum = 0;
		auto energy = Energy_Contributions(spins);
		for (auto E : energy) sum += E.second;
		return sum;
	}

    std::vector<std::pair<std::string, scalar>> Hamiltonian::Energy_Contributions(const vectorfield & spins)
    {
		Energy_Contributions_per_Spin(spins, this->energy_contributions_per_spin);
		std::vector<std::pair<std::string, scalar>> energy(this->energy_contributions_per_spin.size());
		for (unsigned int i = 0; i < energy.size(); ++i)
		{
			energy[i] = { this->energy_contributions_per_spin[i].first, Vectormath::sum(this->energy_contributions_per_spin[i].second) };
		}
		return energy;
    }

	void Hamiltonian::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Contributions_per_Spin() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
    }

	static const std::string name = "--";
    const std::string& Hamiltonian::Name()
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Name() of the Hamiltonian base class!"));
        return name;
    }
}