#include "Optimizer_SIB2.hpp"

#include "utility/Vectormath.hpp"

using namespace Utility;

namespace Engine
{
	Optimizer_SIB2::Optimizer_SIB2(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		//this->virtualforce = std::vector<std::vector<scalar>>(this->noi, std::vector<scalar>(3 * this->nos));	// [noi][3*nos]
		this->spins_temp = std::vector<std::vector<Vector3>>(this->noi, std::vector<Vector3>(this->nos));	// [noi][nos]
		//if (systems.size() > 1) Log(Log_Level::ERROR, Log_Sender::LLG, "THE OPTIMIZER_SIB2 CANNOT HANDLE MORE THAN 1 IMAGE CORRECTLY !!");
    }
	
	void Optimizer_SIB2::Iteration()
	{
		std::shared_ptr<Data::Spin_System> s;

		// Random Numbers
		for (int img = 0; img < this->noi; ++img)
		{
			s = method->systems[img];

			scalar h = s->llg_parameters->dt;
			scalar rh = std::sqrt(h);
			scalar alpha = s->llg_parameters->damping;
			scalar temperature = s->llg_parameters->temperature;

			scalar nx, ny, nz;
			scalar Hx, Hy, Hz;
			scalar Cx, Cy, Cz;
			scalar Rx, Ry, Rz;
			scalar ax, ay, az;
			scalar Ax, Ay, Az;
			scalar detMi;
			scalar D = std::sqrt(2.0*alpha / (1.0 + alpha * alpha) * temperature * Vectormath::kB());

			auto R = std::vector<Vector3>(nos);
			this->Gen_Xi(*s, R, D);

			Cx = s->llg_parameters->stt_polarisation_normal[0] * s->llg_parameters->stt_magnitude;
			Cy = s->llg_parameters->stt_polarisation_normal[1] * s->llg_parameters->stt_magnitude;
			Cz = s->llg_parameters->stt_polarisation_normal[2] * s->llg_parameters->stt_magnitude;

			this->method->Calculate_Force(configurations, force);

			for (int i = 0; i < s->nos; ++i)
			{
				nx = (*s->spins)[i][0]; ny = (*s->spins)[i][1]; nz = (*s->spins)[i][2];
				Hx = force[img][i][0]; Hy = force[img][i][1]; Hz = force[img][i][2];
				Rx = R[i][0]; Ry = R[i][1]; Rz = R[i][2];

				Ax = 0.5*h * (-Hx - alpha * (ny*Hz - nz*Hy));
				Ay = 0.5*h * (-Hy - alpha * (nz*Hx - nx*Hz));
				Az = 0.5*h * (-Hz - alpha * (nx*Hy - ny*Hx));

				Ax = Ax + 0.5*h * (-alpha*Cx + (ny*Cz - nz*Cy));
				Ay = Ay + 0.5*h * (-alpha*Cy + (nz*Cx - nx*Cz));
				Az = Az + 0.5*h * (-alpha*Cz + (nx*Cy - ny*Cx));

				Ax = Ax + 0.5*rh * D * (-Rx - alpha*(ny*Rz - nz*Ry));
				Ay = Ay + 0.5*rh * D * (-Ry - alpha*(nz*Rx - nx*Rz));
				Az = Az + 0.5*rh * D * (-Rz - alpha*(nx*Ry - ny*Rx));

				ax = nx + ny*Az - nz*Ay;
				ay = ny + nz*Ax - nx*Az;
				az = nz + nx*Ay - ny*Ax;

				Hx = Ax*Ax;
				Hy = Ay*Ay;
				Hz = Az*Az;
				Rx = Ay*Az;
				Ry = Ax*Az;
				Rz = Ax*Ay;

				detMi = 1.0 / (1.0 + Hx + Hy + Hz);

				nx = nx + (ax*(1.0 + Hx) + ay*(Rz + Az) + az*(Ry - Ay)) * detMi;
				ny = ny + (ax*(Rz - Az) + ay*(1.0 + Hy) + az*(Rx + Ax)) * detMi;
				nz = nz + (ax*(Ry + Ay) + ay*(Rx - Ax) + az*(1.0 + Hz)) * detMi;

				spins_temp[img][i][0] = 0.5*nx;
				spins_temp[img][i][1] = 0.5*ny;
				spins_temp[img][i][2] = 0.5*nz;
			}

			this->method->Calculate_Force(configurations, force);

			for (int i = 0; i < s->nos; ++i)
			{
				nx = spins_temp[img][i][0]; ny = spins_temp[img][i][1]; nz = spins_temp[img][i][2];
				Hx = force[img][i][0]; Hy = force[img][i][1]; Hz = force[img][i][2];
				Rx = R[i][0]; Ry = R[i][1]; Rz = R[i][2];

				Ax = 0.5*h * (-Hx - alpha * (ny*Hz - nz*Hy));
				Ay = 0.5*h * (-Hy - alpha * (nz*Hx - nx*Hz));
				Az = 0.5*h * (-Hz - alpha * (nx*Hy - ny*Hx));

				Ax = Ax + 0.5*h * (-alpha*Cx + (ny*Cz - nz*Cy));
				Ay = Ay + 0.5*h * (-alpha*Cy + (nz*Cx - nx*Cz));
				Az = Az + 0.5*h * (-alpha*Cz + (nx*Cy - ny*Cx));

				Ax = Ax + 0.5*rh * D * (-Rx - alpha*(ny*Rz - nz*Ry));
				Ay = Ay + 0.5*rh * D * (-Ry - alpha*(nz*Rx - nx*Rz));
				Az = Az + 0.5*rh * D * (-Rz - alpha*(nx*Ry - ny*Rx));

				nx = (*s->spins)[i][0]; ny = (*s->spins)[i][1]; nz = (*s->spins)[i][2];

				ax = nx + ny*Az - nz*Ay;
				ay = ny + nz*Ax - nx*Az;
				az = nz + nx*Ay - ny*Ax;

				Hx = Ax*Ax;
				Hy = Ay*Ay;
				Hz = Az*Az;
				Rx = Ay*Az;
				Ry = Ax*Az;
				Rz = Ax*Ay;

				detMi = 1.0 / (1.0 + Hx + Hy + Hz);

				(*s->spins)[i][0] = (ax*(1.0 + Hx) + ay*(Rz + Az) + az*(Ry - Ay)) * detMi;
				(*s->spins)[i][1] = (ax*(Rz - Az) + ay*(1.0 + Hy) + az*(Rx + Ax)) * detMi;
				(*s->spins)[i][2] = (ax*(Ry + Ay) + ay*(Rx - Ax) + az*(1.0 + Hz)) * detMi;

			}
		}
	}

	void Optimizer_SIB2::Gen_Xi(Data::Spin_System & s, std::vector<Vector3> & xi, scalar eps)
	{
		//for (int i = 0; i < 3*s.nos; ++i) {
		//	// PRNG gives RN int [0,1] -> [-1,1] -> multiply with eps
		//	xi[i] = (s.llg_parameters->distribution_int(s.llg_parameters->prng) * 2 - 1)*eps;
		//}//endfor i
		for (int dim = 0; dim < 3; ++dim) {
			for (int i = 0; i < s.nos; ++i) {
				// PRNG gives RN int [0,1] -> [-1,1] -> multiply with eps
				xi[i][dim] = (s.llg_parameters->distribution_int(s.llg_parameters->prng) * 2 - 1)*eps;
			}//endfor i
		}//enfor dim

	}//end Gen_Xi


    // Optimizer name as string
    std::string Optimizer_SIB2::Name() { return "SIB2"; }
    std::string Optimizer_SIB2::FullName() { return "Semi-implicit B (2nd implementation)"; }
}