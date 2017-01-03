#include <engine/Optimizer_VP.hpp>
#include <engine/Vectormath.hpp>

namespace Engine
{
    Optimizer_VP::Optimizer_VP(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->spins_temp = std::vector<vectorfield>(this->noi, vectorfield(this->nos));	// [noi][nos]
		this->velocity = std::vector<vectorfield>(this->noi, vectorfield(this->nos, Vector3::Zero()));	// [noi][nos]
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
			Vectormath::add_c_a(0.5 / m * dt, l_force_prev, l_velocity);
			Vectormath::add_c_a(0.5 / m * dt, l_force, l_velocity);

			// Get the projection of the velocity on the force
			projection[i] = Vectormath::dot(l_velocity, l_force);
			force_norm2[i] = Vectormath::dot(l_force, l_force);

			// Calculate the projected velocity
			if (projection[i] <= 0)
			{
				Vectormath::fill(l_velocity, { 0,0,0 });
			}
			else
			{
				l_velocity = l_force;
				Vectormath::scale(l_velocity, projection[i] / force_norm2[i]);
			}

			// Copy in
			spins_temp[i] = *(configurations[i]);

			// Move the spins
			Vectormath::add_c_a(dt, l_velocity, spins_temp[i]);
			Vectormath::add_c_a(0.5/m*dt*dt, l_velocity, spins_temp[i]);
			Vectormath::normalize_vectors(spins_temp[i]);

			// Copy out
			*(configurations[i]) = spins_temp[i];
		}
    }
    
    // Optimizer name as string
    std::string Optimizer_VP::Name() { return "VP"; }
    std::string Optimizer_VP::FullName() { return "Velocity Projection"; }
}