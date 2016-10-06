#include <engine/Hamiltonian.hpp>

#include <utility/Vectormath.hpp>

namespace Engine
{
	Hamiltonian::Hamiltonian(std::vector<bool> boundary_conditions) :
        boundary_conditions(boundary_conditions)
    {
        prng = std::mt19937(94199188);
		distribution_int = std::uniform_int_distribution<int>(0, 1);

        delta = 1e-6;
    }

    void Hamiltonian::Hessian(const std::vector<double> & spins, std::vector<double> & hessian)
    {
		// Simultaneous perturbation stochastic approximation of the gradient
		//      (SPSA - see https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation)
		//      in contrast to regular finite differences (FDSA), it needs only 2, instead of 2n evaluations
		//      of the energy.


		int nos = spins.size() / 3;

		// Generate random vector
		std::vector<double> random_vector1(3 * nos, 0);  // inefficient, but nos is not available here...
		std::vector<double> random_vector2(3 * nos, 0);  // inefficient, but nos is not available here...
		for (int dim = 0; dim < 3; ++dim) {
			for (int i = 0; i < nos; ++i) {
				// PRNG gives RN int [0,1] -> [-1,1] -> multiply with delta
				random_vector1[dim*nos + i] = (this->distribution_int(this->prng) * 2 - 1)*delta;
				random_vector2[dim*nos + i] = (this->distribution_int(this->prng) * 2 - 1)*delta;
			}//endfor i
		}//enfor dim

		// Calculate finite difference
		std::vector<double> spins_plus(3 * nos, 0);  // inefficient, but nos is not available here...
		std::vector<double> spins_minus(3 * nos, 0);  // inefficient, but nos is not available here...
		std::vector<double> spins_pp(3 * nos, 0);  // inefficient, but nos is not available here...
		std::vector<double> spins_mp(3 * nos, 0);  // inefficient, but nos is not available here...
		for (int dim = 0; dim < 3; ++dim)
		{
			for (int i = 0; i < nos; ++i)
			{
				spins_plus[i + dim*nos] = spins[i + dim*nos] + random_vector1[i + dim*nos];
				spins_minus[i + dim*nos] = spins[i + dim*nos] - random_vector1[i + dim*nos];
				spins_pp[i + dim*nos] = spins[i + dim*nos] + random_vector1[i + dim*nos] + random_vector2[i + dim*nos];
				spins_mp[i + dim*nos] = spins[i + dim*nos] - random_vector1[i + dim*nos] + random_vector2[i + dim*nos];
			}
		}
		Utility::Vectormath::Normalize_3Nos(spins_plus);
		Utility::Vectormath::Normalize_3Nos(spins_minus);
		Utility::Vectormath::Normalize_3Nos(spins_pp);
		Utility::Vectormath::Normalize_3Nos(spins_mp);
		double E_plus = this->Energy(spins_plus);
		double E_minus = this->Energy(spins_minus);
		double E_pp = this->Energy(spins_pp);
		double E_mp = this->Energy(spins_mp);
		for (int dim = 0; dim < 3; ++dim)
		{
			for (int i = 0; i < nos; ++i)
			{
				for (int j = 0; j < nos; ++j)
				{
					hessian[i*3*nos + dim*nos + j] = (E_pp - E_plus - E_mp + E_minus)
						/ (4 * (random_vector1[i + dim*nos] * random_vector2[j + dim*nos] + random_vector1[j + dim*nos] * random_vector2[i + dim*nos]) );
				}
			}
		}
    }

    void Hamiltonian::Effective_Field(const std::vector<double> & spins, std::vector<double> & field)
    {
        // Simultaneous perturbation stochastic approximation of the gradient
        //      (SPSA - see https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation)
        //      in contrast to regular finite differences (FDSA), it needs only 2, instead of 2n evaluations
        //      of the energy.


        int nos = spins.size()/3;

        // Generate random vector
		std::vector<double> random_vector(3*nos, 0);  // inefficient, but nos is not available here...
        for (int dim = 0; dim < 3; ++dim) {
			for (int i = 0; i < nos; ++i) {
				// PRNG gives RN int [0,1] -> [-1,1] -> multiply with delta
				random_vector[dim*nos + i] = (this->distribution_int(this->prng) * 2 - 1)*delta;
			}//endfor i
		}//enfor dim

        // Calculate finite difference
		std::vector<double> spins_plus(3*nos, 0);  // inefficient, but nos is not available here...
		std::vector<double> spins_minus(3*nos, 0);  // inefficient, but nos is not available here...
        for (int dim = 0; dim < 3; ++dim)
        {
			for (int i = 0; i < nos; ++i)
            {
                spins_plus[i+dim*nos] = spins[i+dim*nos] + random_vector[i+dim*nos];
                spins_minus[i+dim*nos] = spins[i+dim*nos] - random_vector[i+dim*nos];
            }
        }
		Utility::Vectormath::Normalize_3Nos(spins_plus);
		Utility::Vectormath::Normalize_3Nos(spins_minus);
        double E_plus = this->Energy(spins_plus);
        double E_minus = this->Energy(spins_minus);
        for (int dim = 0; dim < 3; ++dim)
        {
			for (int i = 0; i < nos; ++i)
            {
                field[i+dim*nos] = ( E_minus - E_plus) / (2*random_vector[i+dim*nos]);  // Note the order of the Energies (we are calculating eff. field, not gradient)
            }
        }
    }

    double Hamiltonian::Energy(std::vector<double> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return 0.0;
    }

    std::vector<std::vector<double>> Hamiltonian::Energy_Array_per_Spin(std::vector<double> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array_per_Spin() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return std::vector<std::vector<double>>(spins.size(), std::vector<double>(7, 0.0));
    }

    std::vector<double> Hamiltonian::Energy_Array(std::vector<double> & spins)
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Energy_Array() of the Hamiltonian base class!"));
        throw Utility::Exception::Not_Implemented;
        return std::vector<double>(7, 0.0);
    }

    std::string Hamiltonian::Name()
    {
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Hamiltonian::Name() of the Hamiltonian base class!"));
        return "--";
    }
}