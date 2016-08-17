#include "Optimizer_SIB.h"

#include "Vectormath.h"

using namespace Utility;

namespace Engine
{
	Optimizer_SIB::Optimizer_SIB(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->xi = std::vector<double>(3 * this->nos);
		this->virtualforce = std::vector<std::vector<double>>(this->noi, std::vector<double>(3 * this->nos));	// [noi][3*nos]
		
		this->spins_temp = std::vector<std::shared_ptr<std::vector<double>>>(this->noi);
		for (int i=0; i<this->noi; ++i) spins_temp[i] = std::shared_ptr<std::vector<double>>(new std::vector<double>(3 * this->nos)); // [noi][3*nos]
    }

    void Optimizer_SIB::Iteration()
    {
		std::shared_ptr<Data::Spin_System> s;

		// Random Numbers
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			this->epsilon = std::sqrt(2.0*s->llg_parameters->damping / (1.0 + std::pow(s->llg_parameters->damping, 2))*s->llg_parameters->temperature*Vectormath::kB());
			// Precalculate RNs --> move this up into Iterate and add array dimension n for no of iterations?
			this->Gen_Xi(*s, xi, epsilon);
		}

		// First part of the step
		this->method->Calculate_Force(configurations, force);
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			this->VirtualForce(s->nos, *s->spins, *s->llg_parameters, force[i], xi, virtualforce[i]);
			this->FirstStep(s->nos, *s->spins, virtualforce[i], *spins_temp[i]);
		}

		// Second part of the step
		this->method->Calculate_Force(this->spins_temp, force); ////// I cannot see a difference if this step is included or not...
		for (int i = 0; i < this->noi; ++i)
		{
			s = method->systems[i];
			this->VirtualForce(s->nos, *spins_temp[i], *s->llg_parameters, force[i], xi, virtualforce[i]);
			this->SecondStep(s->nos, virtualforce[i], *s->spins);
		}
	}


	void Optimizer_SIB::VirtualForce(const int nos, std::vector<double> & spins, Data::Parameters_LLG & llg_params, std::vector<double> & beff,  std::vector<double> & xi, std::vector<double> & force)
	{
		//========================= Init local vars ================================
		int i, dim;
		// deterministic variables
		double a1[3], b1[3], asc[3];
		// stochastic variables
		double s1[3], f1[3];
		// aux variables
		double A[3];
		// time steps
		double damping = llg_params.damping;
		double sqrtdt = std::sqrt(llg_params.dt), dtg = llg_params.dt, sqrtdtg = sqrtdt;
		// integration variables
		double e1[3];
		// STT
		double a_j = llg_params.stt_magnitude;
		std::vector<double> s_c_vec = llg_params.stt_polarisation_normal;
		//------------------------ End Init ----------------------------------------
		for (i = 0; i < nos; ++i)
		{
			for (dim = 0; dim < 3; ++dim) {
				e1[dim] = spins[dim*nos + i];
				b1[dim] = beff[dim*nos + i];
				f1[dim] = xi[dim*nos + i];
			}
			// Vectormath::Vector_Copy(e1, spins, 3, 0, i);
			// Vectormath::Vector_Copy(b1, beff, 3, 0, i);
			// Vectormath::Vector_Copy(f1, xi, 3, 0, i);

			// a1 = -b1 - damping * (e1 cross b1)
			// change into:
			// a1 = -b1 - damping (e1 cross b1)
			//		-a_j * damping * s_p + a_j * (e1 cross s_p)
			Vectormath::Cross_Product(e1, b1, a1);
			Vectormath::Array_Skalar_Mult(a1, 3, -damping);
			Vectormath::Array_Array_Add(a1, b1, 3, -1.0);
			if (true) {
				Vectormath::Cross_Product(e1, s_c_vec, asc);
				Vectormath::Array_Skalar_Mult(asc, 3, a_j);
				Vectormath::Array_Array_Add(asc, s_c_vec, 3, -a_j*damping);
				Vectormath::Array_Array_Add(a1, asc, 3, 1.0);
			}

			// s1 = -f1 - damping * (e1 cross f1) // s1 is stochastic counterpart of a1
			Vectormath::Cross_Product(e1, f1, s1);
			Vectormath::Array_Skalar_Mult(s1, 3, -damping);
			Vectormath::Array_Array_Add(s1, f1, 3, -1.0);

			/*
			semi - implicitness midpoint requires solution of linear system :
			A*e2 = At*e1, At = transpose(A) = > e2 = inv(A)*At*e1
			A = I + skew(dt*a1 / 2 + sqrt(dt)*s1 / 2)
			write A*e2 = a2, a2 = At*e1 = > e2 = inv(A)*a2
			Ax, Ay, Az off - diagonal components of A
			solve with Cramers' rule => define detAi=1/determinant(A)
			*/

			// ?get h*a_i(X_k) and sqrt(h)*sigma(x_k?)ksi into one expression?
			// A = dtg/2 * a1 + sqrt(dtg)/2 * s1
			Vectormath::Array_Array_Add(a1, s1, A, 3, 0.5*dtg, 0.5*sqrtdtg);

			for (int dim = 0; dim < 3; ++dim)
			{
				force[dim*nos + i] = A[dim];
			}
		}
	}


	void Optimizer_SIB::FirstStep(const int nos, std::vector<double> & spins, std::vector<double> & force, std::vector<double> & spins_temp)
	{

		// aux variables
		double a2[3], A[3], detAi;
		// integration variables
		double e1[3], et[3];
		int dim = 0;

		for (int i = 0; i < nos; ++i)
		{
			for (dim = 0; dim < 3; ++dim) {
				e1[dim] = spins[dim*nos + i];
				A[dim] = force[dim*nos + i];
			}

			// 1/determinant(A)
			detAi = 1.0 / (1 + pow(Vectormath::Length(A, 3), 2.0));

			// calculate equation without the predictor?
			Vectormath::Cross_Product(e1, A, a2);
			Vectormath::Array_Array_Add(a2, e1, 3, 1.0);

			et[0] = (a2[0] * (1 + A[0] * A[0]) + a2[1] * (A[0] * A[1] + A[2]) + a2[2] * (A[0] * A[2] - A[1]))*detAi;
			et[1] = (a2[0] * (A[1] * A[0] - A[2]) + a2[1] * (1 + A[1] * A[1]) + a2[2] * (A[1] * A[2] + A[0]))*detAi;
			et[2] = (a2[0] * (A[2] * A[0] + A[1]) + a2[1] * (A[2] * A[1] - A[0]) + a2[2] * (1 + A[2] * A[2]))*detAi;

			for (dim = 0; dim < 3; ++dim) {
				spins_temp[dim*nos + i] = (e1[dim] + et[dim])*0.5;
			}
		}
	}

	void Optimizer_SIB::SecondStep(const int nos, std::vector<double> & force, std::vector<double> & spins)
	{
		// aux variables
		double a2[3], A[3], detAi;
		// integration variables
		double e1[3], et[3];
		int dim = 0;

		for (int i = 0; i < nos; ++i) {
			for (dim = 0; dim < 3; ++dim) {
				e1[dim] = spins[dim*nos + i];
				A[dim] = force[dim*nos + i];
			}

			// 1/determinant(A)
			detAi = 1.0 / (1 + pow(Vectormath::Length(A, 3), 2.0));

			// calculate equation without the predictor?
			Vectormath::Cross_Product(e1, A, a2);
			Vectormath::Array_Array_Add(a2, e1, 3, 1.0);

			et[0] = (a2[0] * (1 + A[0] * A[0]) + a2[1] * (A[0] * A[1] + A[2]) + a2[2] * (A[0] * A[2] - A[1]))*detAi;
			et[1] = (a2[0] * (A[1] * A[0] - A[2]) + a2[1] * (1 + A[1] * A[1]) + a2[2] * (A[1] * A[2] + A[0]))*detAi;
			et[2] = (a2[0] * (A[2] * A[0] + A[1]) + a2[1] * (A[2] * A[1] - A[0]) + a2[2] * (1 + A[2] * A[2]))*detAi;

			for (dim = 0; dim < 3; ++dim) {
				spins[dim*nos + i] = et[dim];
			}
		}
	}


	void Optimizer_SIB::Gen_Xi(Data::Spin_System & s, std::vector<double> & xi, double eps)
	{
		//for (int i = 0; i < 3*s.nos; ++i) {
		//	// PRNG gives RN int [0,1] -> [-1,1] -> multiply with eps
		//	xi[i] = (s.llg_parameters->distribution_int(s.llg_parameters->prng) * 2 - 1)*eps;
		//}//endfor i
		for (int dim = 0; dim < 3; ++dim) {
			for (int i = 0; i < s.nos; ++i) {
				// PRNG gives RN int [0,1] -> [-1,1] -> multiply with eps
				xi[dim*s.nos + i] = (s.llg_parameters->distribution_int(s.llg_parameters->prng) * 2 - 1)*eps;
			}//endfor i
		}//enfor dim

	}//end Gen_Xi

    // Optimizer name as string
    std::string Optimizer_SIB::Name() { return "SIB"; }
    std::string Optimizer_SIB::FullName() { return "Semi-implicit B"; }
}