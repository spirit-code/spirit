#pragma once
#ifndef METHOD_H
#define METHOD_H

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method.hpp>
#include <utility/Timing.hpp>
#include <utility/Logging.hpp>

#include <deque>
#include <fstream>
#include <map>

namespace Engine
{
    /*
        Abstract Base Class for Simulation/Calculation Methods.
        This class provides the possibility to have pointers to different template instantiations
        of the Method class at runtime. This is needed e.g. to extract information to the State.
    */
    class Method
    {
    public:
        // Constructor to be used in derived classes
        Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain);

        
		// One Iteration
		virtual void Iteration();

		// Iterate for method->parameters->n iterations
		virtual void Iterate() final;

		// Calculate a smooth but current IPS value
		virtual scalar getIterationsPerSecond() final;


        // // Solver_Initialise contains the initialisations of arrays etc. for a certain solver
        // void Solver_Init();
        // // Solver_Step represents one iteration of a certain Solver
        // void Solver_Step();

        void VirtualForce( const vectorfield & spins, 
                                    const Data::Parameters_Method_LLG & llg_params, 
                                    const vectorfield & effective_field,  
                                    const vectorfield & xi, 
                                    vectorfield & force );

        // Check if walltime ran out
        virtual bool Walltime_Expired(duration<scalar> dt_seconds);

        // Calculate Forces onto Systems
        virtual void Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces);
        // Maximum of the absolutes of all components of the force - needs to be updated at each calculation
        scalar force_maxAbsComponent;
        // Check if the Forces are converged
        virtual bool Force_Converged();

        // Check wether to continue iterating - stop file, convergence etc.
        virtual bool ContinueIterating() final;

        // A hook into the Optimizer before an Iteration
        virtual void Hook_Pre_Iteration();
        // A hook into the Optimizer after an Iteration
        virtual void Hook_Post_Iteration();

        // Save the systems' current data
        virtual void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false);
        // Finalize the optimization of the systems
        virtual void Finalize();

        // Lock systems in order to prevent otherwise access
        virtual void Lock();
        // Unlock systems to re-enable access
        virtual void Unlock();

        // Method name as string
        virtual std::string Name();
        // Method name as enum
        Utility::Log_Sender SenderName;

        // Solver name as string
        virtual std::string SolverName();
        virtual std::string SolverFullName();
        
        // // Systems the Optimizer will access
        std::vector<std::shared_ptr<Data::Spin_System>> systems;
        // Method Parameters
        std::shared_ptr<Data::Parameters_Method> parameters;

        // Information for Logging and Save_Current
        int idx_image;
        int idx_chain;
        
    protected:
        // Calculate force_maxAbsComponent for a spin configuration
        virtual scalar Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force) final;
        // Check if iterations_allowed
        virtual bool Iterations_Allowed();
        // History of relevant quantities
        std::map<std::string, std::vector<scalar>> history;

        // Number of Images
        int noi;
        // Number of Spins
        int nos;
        // Number of iterations
        int n_iterations;
        // Number of iterations until log
        int n_iterations_log;
        // Number of times to save
        int n_log;
        // Number of iterations that have been executed
        int iteration;
        // Number of steps (set of iterations between logs) that have been executed
        int step;

        // Pointers to Configurations
        std::vector<std::shared_ptr<vectorfield>> configurations;
        // Actual Forces on the configurations
        std::vector<vectorfield> force;

        // The time at which this Solver's Iterate() was last called
        std::string starttime;
        // Timings and Iterations per Second
        scalar ips;
        std::deque<std::chrono::time_point<std::chrono::system_clock>> t_iterations;

        // Check if a stop file is present -> Stop the iterations
        virtual bool StopFilePresent() final;

        // Precision for the conversion of scalar to string
        int print_precision;
	};
}

#endif