#include <engine/Optimizer_VP.hpp>

namespace Engine
{
    Optimizer_VP::Optimizer_VP(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->spins_temp = std::vector<std::vector<Vector3>>(this->noi, std::vector<Vector3>(this->nos));	// [noi][nos]
		this->velocity = std::vector<std::vector<Vector3>>(this->noi, std::vector<Vector3>(this->nos));	// [noi][nos]
		this->projection = std::vector<scalar>(this->noi, 0);	// [noi]
		this->force_norm2 = std::vector<scalar>(this->noi, 0);	// [noi]
    }

    void Optimizer_VP::Iteration()
    {
		std::shared_ptr<Data::Spin_System> s;

		// Set previous
		force_previous = force;
		velocity_previous = velocity;

		// Get the forces on the configurations
		this->method->Calculate_Force(configurations, force);

		for (int i = 0; i < noi; ++i)
		{
			auto& l_velocity = velocity[i];
			auto& l_force = force[i];
			auto& l_force_prev = force_previous[i];

			s = method->systems[i];
			scalar dt = s->llg_parameters->dt;

			// Calculate the new velocity
			for (int j = 0; j < nos; ++j)
			{
				l_velocity[j] += 0.5 / m * (l_force_prev[j] + l_force[j]) * dt;
			}

			// Get the projection of the velocity on the force
			projection[i] = 0;
			force_norm2[i] = 0;
			for (int j = 0; j < nos; ++j)
			{
				projection[i] += l_velocity[j].dot(l_force[j]);
				force_norm2[i] += l_force[j].norm();
			}

			// Calculate the projected velocity
			if (projection[i] <= 0)
			{
				for (int j = 0; j < nos; ++j)
				{
					l_velocity[j].setZero();
				}
			}
			else
			{
				for (int j = 0; j < nos; ++j)
				{
					l_velocity[j] = projection[i] * l_force[j] / force_norm2[i];
				}
			}

			// Copy in
			spins_temp[i] = *(configurations[i]);

			// Move the spins
			for (int j = 0; j < nos; ++j)
			{
				spins_temp[i][j] += l_velocity[j]*dt + 0.5/m*l_force[j]*dt*dt;
				spins_temp[i][j].normalize();
			}
			// Renormalize
			//Utility::Vectormath::Normalize_3Nos(spins_temp[i]);

			// Copy out
			*(configurations[i]) = spins_temp[i];
		}
    }
    
    // Optimizer name as string
    std::string Optimizer_VP::Name() { return "VP"; }
    std::string Optimizer_VP::FullName() { return "Velocity Projection"; }
}