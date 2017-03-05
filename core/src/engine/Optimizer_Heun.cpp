#include <engine/Optimizer_Heun.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Vectormath.hpp>

#include <Eigen/Dense>


namespace Engine
{
	Optimizer_Heun::Optimizer_Heun(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->virtualforce = std::vector<vectorfield>(this->noi, vectorfield(this->nos));	// [noi][nos][3]
		this->spins_temp = std::vector<vectorfield>(this->noi, vectorfield(this->nos));	// [noi][nos][3]
		this->temp1 = vectorfield(this->nos);	// [nos][3]
		this->temp2 = vectorfield(this->nos);	// [nos][3]
		this->temp3 = vectorfield(this->nos);	// [nos][3]
    }

    void Optimizer_Heun::Iteration()
    {
		std::shared_ptr<Data::Spin_System> s;
		
		// Get the actual forces on the configurations
		this->method->Calculate_Force(configurations, force);

		scalar dt;

		// Optimization for each image
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			auto& conf = *configurations[i];

			dt = s->llg_parameters->dt;

			// Update spins_temp
			Vectormath::set_c_cross(1, conf, force[i], temp3);
			Vectormath::set_c_cross(-100*dt*dt, conf, temp3, temp1);
			Vectormath::set_c_a(1, conf, temp2);
			Vectormath::add_c_a(1, temp1, temp2);
			Vectormath::add_c_cross(1, temp2, force[i], temp3);
			Vectormath::add_c_cross(-100*dt*dt, temp2, temp3, temp1);
			Vectormath::set_c_a(5*dt, conf, spins_temp[i]);
			Vectormath::add_c_a(5*dt, temp1, spins_temp[i]);

			// Normalize spins
			Vectormath::normalize_vectors(spins_temp[i]);

			// Copy out
			conf = spins_temp[i];
		}
	}

    // Optimizer name as string
    std::string Optimizer_Heun::Name() { return "Heun"; }
    std::string Optimizer_Heun::FullName() { return "Heun"; }
}