#include <Optimizer_Heun.hpp>

#include "engine/Hamiltonian.hpp"

#include <Eigen/Dense>


namespace Engine
{
	Optimizer_Heun::Optimizer_Heun(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->virtualforce = std::vector<std::vector<Vector3>>(this->noi, std::vector<Vector3>(this->nos));	// [noi][nos][3]
		this->spins_temp = std::vector<std::vector<Vector3>>(this->noi, std::vector<Vector3>(this->nos));	// [noi][nos][3]
    }

    void Optimizer_Heun::Iteration()
    {
		std::shared_ptr<Data::Spin_System> s;
		
		// Get the actual forces on the configurations
		this->method->Calculate_Force(configurations, force);

		scalar dt;
		Vector3 c1, c2, c3, c4;

		// Optimization for each image
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			auto& conf = *configurations[i];

			Vector3 temp1, temp2;
			dt = s->llg_parameters->dt;

			for (int j = 0; j < nos; ++j)
			{
				c1 = conf[j].cross(force[i][j]);
				c2 = conf[j].cross(c1);
				temp1 = -100 * dt*dt*c2;
				temp2 = conf[j] + temp1;
				c3 = temp2.cross(force[i][j]);
				c4 = temp2.cross(c3);
				temp1 -= 100 * dt*dt*c4;
				spins_temp[i][j] = 5 * dt * (conf[j] + temp1);
				conf[j] = spins_temp[i][j].normalized();
			}
		}
	}

    // Optimizer name as string
    std::string Optimizer_Heun::Name() { return "Heun"; }
    std::string Optimizer_Heun::FullName() { return "Heun"; }
}