#include <Optimizer_Heun.h>

#include "engine/Hamiltonian.h"


namespace Engine
{
	void Optimizer_Heun::Configure(std::vector<std::shared_ptr<Data::Spin_System>> systems, std::shared_ptr<Engine::Force> force_call)
	{
		Optimizer::Configure(systems, force_call);

		this->virtualforce = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos));	// [noi][3*nos]
		this->spins_temp = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos));	// [noi][3*nos]
	}
    
    void Optimizer_Heun::Step()
    {
		std::shared_ptr<Data::Spin_System> s;
		
		// Get the actual forces on the configurations
		this->force_call->Calculate(configurations, force);

		int dim;
		double dt, s2;
		std::vector <double> c1(3), c2(3), c3(3), c4(3);

		// Optimization for each image
		for (int i = 0; i < this->noi; ++i)
		{
			s = systems[i];

			std::vector<double> temp1(3), temp2(3);
			dt = s->llg_parameters->dt;

			for (int j = 0; j < nos; ++j)
			{
				
				for (dim = 0; dim < 3; ++dim)
				{
					c1[dim] = configurations[i][((dim + 1) % 3)*nos + j] * force[i][((dim + 2) % 3)*nos + j]
							- configurations[i][((dim + 2) % 3)*nos + j] * force[i][((dim + 1) % 3)*nos + j];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					c2[dim] = configurations[i][((dim + 1) % 3)*nos + j] * c1[((dim + 2) % 3)]
							- configurations[i][((dim + 2) % 3)*nos + j] * c1[((dim + 1) % 3)];
				}
				for (dim = 0; dim < 3; ++dim)
				{
					temp1[dim] = -100*dt*dt*c2[dim];
					temp2[dim] = configurations[i][dim*nos + j] + temp1[dim];
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
					configurations[i][j + dim*nos] = 10*dt / 2.0 * (configurations[i][j + dim*nos] + temp1[dim]);
				}
				// Renormalize Spins
				s2 = 0;
				for (dim = 0; dim < 3; ++dim)
				{
					s2 += std::pow(configurations[i][j + dim*nos], 2);
				}
				s2 = std::sqrt(s2);
				for (dim = 0; dim < 3; ++dim)
				{
					configurations[i][j + dim*nos] = configurations[i][j + dim*nos] / s2;
				}
			}
			s->spins = configurations[i];
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
    std::string Optimizer_Heun::Name() { return "CG"; }
    std::string Optimizer_Heun::Fullname() { return "Conjugate Gradient"; }
}