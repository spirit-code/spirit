#pragma once
#ifndef METHOD_H
#define METHOD_H

#include "Parameters_Method.h"
#include "Spin_System_Chain.h"
#include "Parameters_Method.h"
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
		Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain);

		// Calculate Forces onto Systems
		virtual void Calculate_Force(std::vector<std::shared_ptr<std::vector<double>>> configurations, std::vector<std::vector<double>> & forces);

		// Check if the Forces are converged
		virtual bool Force_Converged();

		// Maximum of the absolutes of all components of the force - needs to be updated at each calculation
		double force_maxAbsComponent;

		// Method name as string
		virtual std::string Name();
		// Method name as enum
		Utility::Log_Sender SenderName;
		
		// Save the current Step's Data
		virtual void Save_Step(int iteration, bool final);
		// A hook into the Optimizer before an Iteration
		virtual void Hook_Pre_Step();
		// A hook into the Optimizer after an Iteration
		virtual void Hook_Post_Step();

		// Systems the Optimizer will access
		std::vector<std::shared_ptr<Data::Spin_System>> systems;

		// Method Parameters
		std::shared_ptr<Data::Parameters_Method> parameters;

		// Check wether to continue iterating - stop file, convergence etc.
		virtual bool ContinueIterating() final;

	protected:
		// Information for Save_Step
		int idx_image;
		int idx_chain;
		// Calculate force_maxAbsComponent for a spin configuration
		virtual double Force_on_Image_MaxAbsComponent(const std::vector<double> & image, std::vector<double> force) final;
		
	};
}

#endif