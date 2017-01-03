#include <engine/Optimizer_SIB.hpp>
#include <engine/Vectormath.hpp>

namespace Engine
{
	Optimizer_SIB::Optimizer_SIB(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		this->xi = vectorfield(this->nos);
		this->virtualforce = std::vector<vectorfield>(this->noi, vectorfield(this->nos));	// [noi][nos]
		
		this->spins_temp = std::vector<std::shared_ptr<vectorfield>>(this->noi);
		for (int i=0; i<this->noi; ++i) spins_temp[i] = std::shared_ptr<vectorfield>(new vectorfield(this->nos)); // [noi][nos]
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


	void Optimizer_SIB::VirtualForce(const int nos, vectorfield & spins, Data::Parameters_Method_LLG & llg_params, vectorfield & eff_field,  vectorfield & xi, vectorfield & force)
	{
		//========================= Init local vars ================================
		int i;
		// deterministic variables
		Vector3 a1, b1, asc;
		// stochastic variables
		Vector3 s1, f1;
		// aux variables
		Vector3 A;
		// time steps
		scalar damping = llg_params.damping;
		scalar sqrtdt = std::sqrt(llg_params.dt), dtg = llg_params.dt, sqrtdtg = sqrtdt;
		// integration variables
		Vector3 e1;
		// STT
		scalar a_j = llg_params.stt_magnitude;
		Vector3 s_c_vec = llg_params.stt_polarisation_normal;
		//------------------------ End Init ----------------------------------------

		Vectormath::fill       (force, {0,0,0});
		Vectormath::add_c_a    (-0.5 * dtg, eff_field, force);
		Vectormath::add_c_cross(-0.5 * dtg * damping, spins, eff_field, force);
		Vectormath::add_c_a    ( 0.5 * dtg * a_j * damping, s_c_vec, force);
		Vectormath::add_c_cross(-0.5 * dtg * a_j, s_c_vec, spins, force);

		Vectormath::add_c_a    (-0.5 * sqrtdtg, xi, force);
		Vectormath::add_c_cross(-0.5 * sqrtdtg * damping, spins, xi, force);
	}


	void Optimizer_SIB::FirstStep(const int nos, vectorfield & spins, vectorfield & force, vectorfield & spins_temp)
	{
		// aux variables
		Vector3 a2, A;
		scalar detAi;
		// integration variables
		Vector3 e1, et;
		int dim = 0;

		for (int i = 0; i < nos; ++i)
		{
			e1 = spins[i];
			A = force[i];

			// 1/determinant(A)
			detAi = 1.0 / (1 + pow(A.norm(), 2.0));

			// calculate equation without the predictor?
			a2 = e1 + e1.cross(A);

			et[0] = (a2[0] * (1 + A[0] * A[0])    + a2[1] * (A[0] * A[1] + A[2]) + a2[2] * (A[0] * A[2] - A[1]))*detAi;
			et[1] = (a2[0] * (A[1] * A[0] - A[2]) + a2[1] * (1 + A[1] * A[1])    + a2[2] * (A[1] * A[2] + A[0]))*detAi;
			et[2] = (a2[0] * (A[2] * A[0] + A[1]) + a2[1] * (A[2] * A[1] - A[0]) + a2[2] * (1 + A[2] * A[2]))*detAi;

			spins_temp[i] = (e1 + et)*0.5;
		}
	}

	void Optimizer_SIB::SecondStep(const int nos, vectorfield & force, vectorfield & spins)
	{
		// aux variables
		Vector3 a2, A;
		scalar detAi;
		// integration variables
		Vector3 e1, et;
		int dim = 0;

		for (int i = 0; i < nos; ++i) {

			e1 = spins[i];
			A = force[i];

			// 1/determinant(A)
			detAi = 1.0 / (1 + pow(A.norm(), 2.0));

			// calculate equation without the predictor?
			a2 = e1 + e1.cross(A);

			et[0] = (a2[0] * (1 + A[0] * A[0]) + a2[1] * (A[0] * A[1] + A[2]) + a2[2] * (A[0] * A[2] - A[1]))*detAi;
			et[1] = (a2[0] * (A[1] * A[0] - A[2]) + a2[1] * (1 + A[1] * A[1]) + a2[2] * (A[1] * A[2] + A[0]))*detAi;
			et[2] = (a2[0] * (A[2] * A[0] + A[1]) + a2[1] * (A[2] * A[1] - A[0]) + a2[2] * (1 + A[2] * A[2]))*detAi;

			spins[i] = et;
		}
	}


	void Optimizer_SIB::Gen_Xi(Data::Spin_System & s, vectorfield & xi, scalar eps)
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
    std::string Optimizer_SIB::Name() { return "SIB"; }
    std::string Optimizer_SIB::FullName() { return "Semi-implicit B"; }
}