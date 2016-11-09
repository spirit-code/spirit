#include <Optimizer_Heun.hpp>

#include "engine/Hamiltonian.hpp"


namespace Engine
{
	Optimizer_Heun::Optimizer_Heun(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->virtualforce = std::vector<std::vector<scalar>>(this->noi, std::vector<scalar>(3 * this->nos));	// [noi][3*nos]
		this->spins_temp = std::vector<std::vector<scalar>>(this->noi, std::vector<scalar>(3 * this->nos));	// [noi][3*nos]
    }

    void Optimizer_Heun::Iteration()
    {
		std::shared_ptr<Data::Spin_System> s;
		
		// Get the actual forces on the configurations
		this->method->Calculate_Force(configurations, force);

		int dim;
		scalar dt, s2;
		std::vector <scalar> c1(3), c2(3), c3(3), c4(3);

		// Optimization for each image
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			auto& conf = *configurations[i];

			std::vector<scalar> temp1(3), temp2(3);
			dt = s->llg_parameters->dt;

			for (int j = 0; j < nos; ++j)
			{
				
				for (dim = 0; dim < 3; ++dim)
				{
					c1[dim] = conf[((dim + 1) % 3)*nos + j] * force[i][((dim + 2) % 3)*nos + j]
							- conf[((dim + 2) % 3)*nos + j] * force[i][((dim + 1) % 3)*nos + j];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					c2[dim] = conf[((dim + 1) % 3)*nos + j] * c1[((dim + 2) % 3)]
							- conf[((dim + 2) % 3)*nos + j] * c1[((dim + 1) % 3)];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					temp1[dim] = -100*dt*dt*c2[dim];
					temp2[dim] = conf[dim*nos + j] + temp1[dim];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					c3[dim] = temp2[((dim + 1) % 3)] * force[i][((dim + 2) % 3)*nos + j]
							- temp2[((dim + 2) % 3)] * force[i][((dim + 1) % 3)*nos + j];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					c4[dim] = temp2[((dim + 1) % 3)] * c3[((dim + 2) % 3)]
							- temp2[((dim + 2) % 3)] * c3[((dim + 1) % 3)];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					temp1[dim] = temp1[dim] - 100*dt*dt*c4[dim];
					spins_temp[i][j + dim*nos] = 10*dt / 2.0 * (conf[j + dim*nos] + temp1[dim]);
				}
				// Renormalize Spins
				s2 = 0;
				for (dim = 0; dim < 3; ++dim)
				{
					s2 += std::pow(spins_temp[i][j + dim*nos], 2);
				}
				s2 = std::sqrt(s2);
				for (dim = 0; dim < 3; ++dim)
				{
					conf[j + dim*nos] = spins_temp[i][j + dim*nos] / s2;
				}
			}
			// s->spins = conf;
			/*
			do i=1,NOS
				temp1 = -dt*dt*cross_product( IMAGES(idx_img,i,:),cross_product(IMAGES(idx_img,i,:), H_eff_tot(i,:)) ) !! relaxation_factor instead of dt*dt?
				temp2 = IMAGES(idx_img,i,:)+temp1
				temp1 = temp1-dt*dt*cross_product( temp2,cross_product(temp2, H_eff_tot(i,:)) ) !! relaxation_factor instead of dt*dt?
				IMAGES(idx_img,i,:) = dt/2_8*(IMAGES(idx_img,i,:)+temp1(:))
			enddo
			*/
		}
	}

    // Optimizer name as string
    std::string Optimizer_Heun::Name() { return "Heun"; }
    std::string Optimizer_Heun::FullName() { return "Heun"; }
}