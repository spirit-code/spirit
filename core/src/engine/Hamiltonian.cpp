#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath.hpp>

namespace Engine
{
	Hamiltonian::Hamiltonian(std::vector<bool> boundary_conditions) :
        boundary_conditions(boundary_conditions)
    {
        prng = std::mt19937(94199188);
		distribution_int = std::uniform_int_distribution<int>(0, 1);

        delta = 1e-6;
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

	void Hamiltonian::Hessian(const std::vector<Vector3> & spins, MatrixX & hessian)
	{
		// This is a regular finite difference implementation (probably not very efficient)
		// using the differences between gradient values (not function)
		// see https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

		int nos = spins.size() / 3;

		// Calculate finite difference
		std::vector<Vector3> spins_p(nos);
		std::vector<Vector3> spins_m(nos);

		std::vector<std::vector<Vector3>> grad_p(3*nos, std::vector<Vector3>(nos));
		std::vector<std::vector<Vector3>> grad_m(3*nos, std::vector<Vector3>(nos));

		std::vector<scalar> d(3 * nos);

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

				d[i + dim*nos] = Vectormath::dist_greatcircle(spins_m[i], spins_p[i]);
				//d[i + dim*nos] = Utility::Manifoldmath::Dist_Geodesic(spins_m, spins_p);
				if (d[i + dim*nos] > 0)
				{
					this->Effective_Field(spins_p, grad_p[i + dim*nos]);
					this->Effective_Field(spins_m, grad_m[i + dim*nos]);
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

    void Hamiltonian::Effective_Field(const std::vector<Vector3> & spins, std::vector<Vector3> & field)
    {
		// This is a regular finite difference implementation (probably not very efficient)

        int nos = spins.size()/3;

		// Calculate finite difference
		std::vector<Vector3> spins_plus(nos);
		std::vector<Vector3> spins_minus(nos);

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

				scalar d = Vectormath::dist_greatcircle(spins_minus[i], spins_plus[i]);

				if (d > 0)
				{
					scalar E_plus = this->Energy(spins_plus);
					scalar E_minus = this->Energy(spins_minus);
					field[i][dim] = (E_minus - E_plus) / d;
				}
				else field[i][dim] = 0;
			}
		}
    }

    scalar Hamiltonian::Energy(const std::vector<Vector3> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return 0.0;
    }

    std::vector<std::vector<scalar>> Hamiltonian::Energy_Array_per_Spin(const std::vector<Vector3> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array_per_Spin() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return std::vector<std::vector<scalar>>(spins.size(), std::vector<scalar>(7, 0.0));
    }

    std::vector<scalar> Hamiltonian::Energy_Array(const std::vector<Vector3> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return std::vector<scalar>(7, 0.0);
    }

	static const std::string name = "--";
    const std::string& Hamiltonian::Name()
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Name() of the Hamiltonian base class!"));
        return name;
    }
}