#include <Spirit_Defines.h>
#include <engine/Method_MC.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/IO.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <ctime>
#include <math.h>

using namespace Utility;

namespace Engine
{
	template <Solver solver>
    Method_MC<solver>::Method_MC(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
		Method_Template<solver>(system->mc_parameters, idx_img, idx_chain)
	{
		// Currently we only support a single image being iterated at once:
		this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
		this->SenderName = Utility::Log_Sender::MC;

		// We assume it is not converged before the first iteration
		this->force_maxAbsComponent = system->mc_parameters->force_convergence + 1.0;

		// History
        this->history = std::map<std::string, std::vector<scalar>>{
			{"max_torque_component", {this->force_maxAbsComponent}},
			{"E", {this->force_maxAbsComponent}},
			{"M_z", {this->force_maxAbsComponent}} };
	}


	template <Solver solver>
	void Method_MC<solver>::Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces)
	{

	}


	template <Solver solver>
	bool Method_MC<solver>::Force_Converged()
	{
		return false;
	}

	template <Solver solver>
	void Method_MC<solver>::Hook_Pre_Iteration()
    {

	}

	template <Solver solver>
    void Method_MC<solver>::Hook_Post_Iteration()
    {
		
    }

	template <Solver solver>
	void Method_MC<solver>::Finalize()
    {
		
    }

	
	template <Solver solver>
	void Method_MC<solver>::Save_Current(std::string starttime, int iteration, bool initial, bool final)
	{
		
	}

	// Optimizer name as string
	template <Solver solver>
    std::string Method_MC<solver>::Name() { return "MC"; }
}