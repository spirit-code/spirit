#include <engine/Optimizer_Heun.hpp>
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
		this->spins_predictor = vectorfield(this->nos);	// [nos][3]
		this->temp2 = vectorfield(this->nos);	// [nos][3]
    }

    void Optimizer_Heun::Iteration()
    {
		std::shared_ptr<Data::Spin_System> s;
		
		// Get the actual forces on the configurations
		this->method->Calculate_Force(configurations, force);

		// Optimization for each image
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			auto& conf = *configurations[i];

			// Scaling to similar stability range as SIB
			scalar dt = s->llg_parameters->dt/10;

			// Predictor
			Vectormath::set_c_cross(1, conf, force[i], temp2);
			Vectormath::set_c_cross(-dt, conf, temp2, temp1);
			Vectormath::set_c_a(1, conf, spins_predictor);
			Vectormath::add_c_a(1, temp1, spins_predictor);

			// Corrector
			Vectormath::add_c_cross(1, spins_predictor, force[i], temp2);
			Vectormath::add_c_cross(-dt, spins_predictor, temp2, temp1);
			Vectormath::set_c_a(0.5*dt, conf, spins_temp[i]);
			Vectormath::add_c_a(0.5*dt, temp1, spins_temp[i]);

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