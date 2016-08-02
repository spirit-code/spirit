#pragma once
#ifndef METHOD_H
#define METHOD_H

#include "Parameters_Method.h"
#include "Spin_System_Chain.h"
#include "Timing.h"

#include <deque>
#include <fstream>

namespace Engine
{
	/*
		Base Class for Methods
	*/
	class Method
	{
	public:
		// Constructor to be used in derived classes
		Method(std::shared_ptr<Data::Parameters_Method> parameters);

		// Calculate Forces onto Systems
		virtual void Calculate_Force(std::vector<std::vector<double>> configurations, std::vector<std::vector<double>> & forces);

		// Check if the Forces are converged
		virtual bool Force_Converged();

		// Maximum of the absolutes of all components of the force - needs to be updated at each calculation
		double force_maxAbsComponent;

		// Method name as string
		virtual std::string Name();
		// TODO: Method name as enum
		//Utility::Log_Sender SenderName;
		
		// Save the current Step's Data
		virtual void Save_Step(int image, int iteration, std::string suffix);
		// A hook into the Optimizer before an Iteration
		virtual void Hook_Pre_Step();
		// A hook into the Optimizer after an Iteration
		virtual void Hook_Post_Step();

		// This Method's Parameters
		std::shared_ptr<Data::Parameters_Method> parameters; // TODO: It would be preferable to have these as protected
	
	protected:
		virtual double Force_on_Image_MaxAbsComponent(std::vector<double> & image, std::vector<double> force) final;
		// The Images to operate on
		// std::vector<std::shared_ptr<Data::Spin_System>> systems;

		//// Create the Force specific to the Solver
		//virtual void Configure()
		//{
		//	this->force_call = std::shared_ptr<Force>(new Force(c));
		//	// Not Implemented!
		//	Utility::Log.Send(Utility::Log_Level::L_ERROR, Utility::Log_Sender::ALL, std::string("Tried to use Solver::Configure() of the Solver base class!"));
		//	//throw Utility::Exception::Not_Implemented;
		//}

	};
}

#endif