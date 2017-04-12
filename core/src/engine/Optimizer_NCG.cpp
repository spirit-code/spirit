#include <engine/Optimizer_NCG.hpp>


namespace Engine
{
    Optimizer_NCG::Optimizer_NCG(std::shared_ptr<Engine::Method> method) :
        Optimizer(method)
    {
		// int jmax = 500;

		// double err_NR = 1e-5;	// Newton-Raphson error threshold

		// double delta_0, delta_new, delta_d;
		// double alpha;
		// r = -f' = eff_field // this->method->Calculate_Force(configurations, force);
		// delta_0 = r*r
		// delta_new = delta_0
		// beta = 0
		// d = r //+ beta*d
    }

    void Optimizer_NCG::Iteration()
    {
        // Not Implemented!
        Log(Utility::Log_Level::Error, Utility::Log_Sender::All, std::string("Tried to use Optimizer::Step() of the Optimizer base class!"));

		
		// Perform a Newton-Raphson line search in order to determine the minimum along d
		// delta_d = d*d
		// for (int j = 0; j < jmax && std::pow(alpha,2)*delta_d > std::pow(err_NR,2); ++j)
		// {
			// alpha = - (f'*d)/(d*f''*d)	// ToDo: How to get the second derivative from here??
			// x = x + alpha*d
		// }

		// Update the direction d
		// r = -f' = eff_field
		// delta_old = delta_new
		// delta_new = r*r
		// beta = delta_new/delta_old
		// d = r + beta*d

		// Restart if d is not a descent direction or after nos iterations
		//		The latter improves convergence for small nos
		// ++k
		// if (r*d <= 0 || k==nos)
		//{
			// d = r
			// k = 0
		//}
    }

    // Optimizer name as string
    std::string Optimizer_NCG::Name() { return "NCG"; }
    std::string Optimizer_NCG::FullName() { return "Nonlinear Conjugate Gradient"; }
}