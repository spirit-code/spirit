#include "Optimizer_VP.h"

#include "Vectormath.h"

namespace Engine
{
    Optimizer_VP::Optimizer_VP(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->spins_temp = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos));	// [noi][3*nos]
		this->velocity = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos, 0));	// [noi][3*nos]
		this->projection = std::vector<double>(this->noi, 0);	// [noi]
		this->force_norm2 = std::vector<double>(this->noi, 0);	// [noi]
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
			double dt = s->llg_parameters->dt;

			// Calculate the new velocity
			for (int j = 0; j < 3 * nos; ++j)
			{
				l_velocity[j] += 0.5 / m * (l_force_prev[j] + l_force[j]) * dt;
			}

			// Get the projection of the velocity on the force
			projection[i] = 0;
			force_norm2[i] = 0;
			for (int j = 0; j < 3*nos; ++j)
			{
				projection[i] += l_velocity[j] * l_force[j];
				force_norm2[i] += l_force[j] * l_force[j];
			}

			// Calculate the projected velocity
			if (projection[i] <= 0)
			{
				for (int j = 0; j < 3 * nos; ++j)
				{
					l_velocity[j] = 0;
				}
			}
			else
			{
				for (int j = 0; j < 3 * nos; ++j)
				{
					l_velocity[j] = projection[i] * l_force[j] / force_norm2[i];
				}
			}

			// Copy in
			spins_temp[i] = *(configurations[i]);

			// Move the spins
			for (int j = 0; j < 3 * nos; ++j)
			{
				spins_temp[i][j] += l_velocity[j]*dt + 0.5/m*l_force[j]*dt*dt;
			}
			// Renormalize
			Utility::Vectormath::Normalize_3Nos(spins_temp[i]);

			// Copy out
			*(configurations[i]) = spins_temp[i];
		}
    }
    
    // Optimizer name as string
    std::string Optimizer_VP::Name() { return "VP"; }
    std::string Optimizer_VP::FullName() { return "Velocity Projection"; }
}