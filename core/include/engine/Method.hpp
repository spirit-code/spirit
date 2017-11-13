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
        //      The constructor does not allocate any large arrays. The solvers should allocate
        //      what they need in Solver_Initialise.
        Method(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain);

        // `Iterate` is supposed to iteratively solve a problem
        virtual void Iterate();

        // Calculate a smooth but current IPS value
        virtual scalar getIterationsPerSecond() final;

        // Maximum of the absolutes of all components of the force - needs to be updated at each calculation
        virtual scalar getForceMaxAbsComponent() final;

        // Maximum of the absolutes of all components of the force for all images the method uses
        // The default is that this returns simply {getForceMaxAbsComponent()}
        virtual std::vector<scalar> getForceMaxAbsComponent_All();

        // Method name as string
        virtual std::string Name();

        // Solver name as string
        virtual std::string SolverName();
        virtual std::string SolverFullName();


    protected:
        // One iteration of the Method
        virtual void Iteration();


        // Initialize arrays etc. before `Iterate` starts
        virtual void Initialize();
        // Finalize (e.g. output informaiton) after `Iterate` has finished
        virtual void Finalize();

        // Log messages at Start, after Steps and at End of Iterate
        virtual void Message_Start();
        virtual void Message_Step();
        virtual void Message_End();

        // A hook into `Iterate` before an Iteration.
        //      Override this function if special actions are needed
        //      before each `Solver_Iteration`
        virtual void Hook_Pre_Iteration();
        // A hook into `Iterate` after an Iteration.
        //      Override this function if special actions are needed
        //      after each `Solver_Iteration`
        virtual void Hook_Post_Iteration();


        // Save current data
        //      Override to specialize what a Method should save
        virtual void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false);


        // Lock systems in order to prevent otherwise access
        //      This function should be overridden by specialized methods to ensure systems are
        //      safely locked during iterations.
        virtual void Lock();
        // Unlock systems to re-enable access
        //      This function should be overridden by specialized methods to ensure systems are
        //      correctly unlocked after iterations.
        virtual void Unlock();




        //////////// Check for stopping criteria //////////////////////////////////////////

        // Check if iterations allowed
        virtual bool Iterations_Allowed();

        // Check if convergence criteria have been met
        virtual bool Converged();

        //////////// Final implementations
        // Check wether to continue iterating - stop file, convergence etc.
        virtual bool ContinueIterating() final;
        // Check if walltime ran out
        virtual bool Walltime_Expired(duration<scalar> dt_seconds) final;
        // Check if a stop file is present -> Stop the iterations
        virtual bool StopFile_Present() final;


        //////////// Information for Logging and Save_Current ////////////////////////
        int idx_image;
        int idx_chain;

        // Number of images
        int noi;
        // Number of spins in an image
        int nos;

        // Number of iterations that have been executed
        int iteration;
        // Number of steps (set of iterations between logs) that have been executed
        int step;

        // Method name as enum
        Utility::Log_Sender SenderName;
        // Maximum force component of all images
        scalar force_max_abs_component;
        // Maximum force component per image
        std::vector<scalar> force_max_abs_component_all;
        // History of relevant quantities
        std::map<std::string, std::vector<scalar>> history;
        // The time at which this Solver's Iterate() was last called
        std::string starttime;
        // Timings and Iterations per Second
        scalar ips;
        std::deque<std::chrono::time_point<std::chrono::system_clock>> t_iterations;
        
        std::chrono::time_point<std::chrono::system_clock> t_start, t_last;

        //////////// Parameters //////////////////////////////////////////////////////
        // Number of iterations
        int n_iterations;
        // Number of iterations until log
        int n_iterations_log;
        // Number of times to save
        int n_log;


        //////////// General /////////////////////////////////////////////////////////

        // Systems the Solver will access
        std::vector<std::shared_ptr<Data::Spin_System>> systems;

        // Method Parameters
        std::shared_ptr<Data::Parameters_Method> parameters;
        
        // Actual Forces on the configurations
        std::vector<vectorfield> forces;
        std::vector<vectorfield> forces_predictor;
        // Virtual Forces used in the Steps
        std::vector<vectorfield> forces_virtual;
        std::vector<vectorfield> forces_virtual_predictor;
        // Random vector array
        vectorfield xi;
        
        // Temporary Spins arrays
        vectorfield temp1, temp2;

        // Pointers to Configurations (for Solver methods)
        std::vector<std::shared_ptr<vectorfield>> configurations;
        std::vector<std::shared_ptr<vectorfield>> configurations_predictor;
        std::vector<std::shared_ptr<vectorfield>> configurations_temp;

        // Precision for the conversion of scalar to string
        int print_precision;
	};
}

#endif