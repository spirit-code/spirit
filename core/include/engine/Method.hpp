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

        // Iterate for method->parameters->n iterations
        virtual void Iterate() final;

        // Calculate a smooth but current IPS value
        virtual scalar getIterationsPerSecond() final;

        // Maximum of the absolutes of all components of the force - needs to be updated at each calculation
        virtual scalar getForceMaxAbsComponent() final;

        // Method name as string
        virtual std::string Name();

        // Solver name as string
        virtual std::string SolverName();
        virtual std::string SolverFullName();


    protected:
        // Solver_Initialise contains the initialisations of arrays etc. for a certain solver
        virtual void Solver_Initialise();
        // Solver_Iteration represents one iteration of a certain Solver
        virtual void Solver_Iteration();



        // Calculate Forces onto Systems
        //      This is currently overridden by methods to specify how the forces on a set of configurations should be
        //      calculated. This function is used in `the Solver_...` functions.
        // TODO: this should not use `vectod<shared_ptr<...>>`, instead `const vector<vectorfield>> &`
        // TODO: maybe rename to separate from deterministic and stochastic force functions
        virtual void Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces);

        // Calculate the effective force on a configuration. It is a combination of
        //      precession and damping terms for the Hamiltonian, spin currents and
        //      temperature. This function is used in `the Solver_...` functions.
        // TODO: separate out the temperature term and create Force_Deterministic and Force_Stochastic
        virtual void VirtualForce( const vectorfield & spins, 
                                    const Data::Parameters_Method_LLG & llg_params, 
                                    const vectorfield & effective_field,  
                                    const vectorfield & xi, 
                                    vectorfield & force ) final;


        // Lock systems in order to prevent otherwise access
        //      This function should be overridden by specialized methods to ensure systems are
        //      safely locked during iterations.
        virtual void Lock();
        // Unlock systems to re-enable access
        //      This function should be overridden by specialized methods to ensure systems are
        //      correctly unlocked after iterations.
        virtual void Unlock();


        // A hook into `Iterate` before an Iteration.
        //      Override this function if special actions are needed
        //      before each `Solver_Iteration`
        virtual void Hook_Pre_Iteration();
        // A hook into `Iterate` after an Iteration.
        //      Override this function if special actions are needed
        //      after each `Solver_Iteration`
        virtual void Hook_Post_Iteration();

        // Finalize the optimization of the systems
        virtual void Finalize();

        // Save the systems' current data
        //      Override to specialize what a Method should save
        virtual void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false);



        // Check wether to continue iterating - stop file, convergence etc.
        virtual bool ContinueIterating() final;
        // Check if iterations_allowed
        virtual bool Iterations_Allowed();
        // Check if the Forces are converged
        virtual bool Force_Converged();
        // Check if walltime ran out
        virtual bool Walltime_Expired(duration<scalar> dt_seconds) final;
        // Check if a stop file is present -> Stop the iterations
        virtual bool StopFile_Present() final;

        // Calculate force_maxAbsComponent for a spin configuration
        virtual scalar Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force) final;



        //////////// Information for Logging and Save_Current //////////////////////////////////////////
        int idx_image;
        int idx_chain;

        // Number of Images
        int noi;
        // Number of Spins
        int nos;

        // Number of iterations that have been executed
        int iteration;
        // Number of steps (set of iterations between logs) that have been executed
        int step;

        //////////// Parameters //////////////////////////////////////////
        // Number of iterations
        int n_iterations;
        // Number of iterations until log
        int n_iterations_log;
        // Number of times to save
        int n_log;




        //////////// General //////////////////////////////////////////

        // Method name as enum
        Utility::Log_Sender SenderName;
        // 
        scalar force_maxAbsComponent;
        // History of relevant quantities
        std::map<std::string, std::vector<scalar>> history;


        // Systems the Optimizer will access
        std::vector<std::shared_ptr<Data::Spin_System>> systems;
        // Method Parameters
        std::shared_ptr<Data::Parameters_Method> parameters;

        // Pointers to Configurations
        std::vector<std::shared_ptr<vectorfield>> configurations;
        // Actual Forces on the configurations
        std::vector<vectorfield> force;

        // The time at which this Solver's Iterate() was last called
        std::string starttime;
        // Timings and Iterations per Second
        scalar ips;
        std::deque<std::chrono::time_point<std::chrono::system_clock>> t_iterations;

        // Temporary Spins arrays
        std::vector<std::shared_ptr<vectorfield>> spins_temp;
        std::vector<std::shared_ptr<vectorfield>> spins_predictor;
        vectorfield temp1, temp2;
        
        // Random vector array
        vectorfield xi;
        // Some variable
        scalar epsilon;

        // pointer to spin system
        std::shared_ptr<Data::Spin_System> s;

        // method time step
        scalar dt;
        scalar dtg;
        // Virtual Forces used in the Steps
        std::vector<vectorfield> virtualforce;
        std::vector<vectorfield> virtualforce_predictor;
        std::vector<vectorfield> rotationaxis;
        std::vector<scalarfield> virtualforce_norm;
        
        // preccession angle
        scalarfield angle;

        // Precision for the conversion of scalar to string
        int print_precision;

        //////////// NCG //////////////////////////////////////////
        // check if the Newton-Raphson has converged
        bool NR_converged();
        
        int jmax;     // max iterations for Newton-Raphson loop
        int n;        // number of iteration after which the nCG will restart
        
        scalar tol_nCG, tol_NR;   // tolerances for optimizer and Newton-Raphson
        scalar eps_nCG, eps_NR;   // Newton-Raphson and optimizer tolerance squared
        
        bool restart_nCG, continue_NR;  // conditions for restarting nCG or continuing Newton-Raphson 
        
        // step sizes
        std::vector<scalarfield> alpha, beta;
        
        // XXX: right type might be std::vector<scalar> and NOT std::vector<scalarfield>
        // delta scalarfields
        std::vector<scalarfield> delta_0, delta_new, delta_old, delta_d;
        
        // residual and new configuration states
        std::vector<vectorfield> res, d;

        // buffer variables for checking convergence for optimizer and Newton-Raphson
        std::vector<scalarfield> r_dot_d, dda2;

        //////////// VP //////////////////////////////////////////
        // "Mass of our particle" which we accelerate
        scalar m = 1.0;

        // Force in previous step [noi][nos]
        std::vector<vectorfield> force_previous;
        // Velocity in previous step [noi][nos]
        std::vector<vectorfield> velocity_previous;
        // Velocity used in the Steps [noi][nos]
        std::vector<vectorfield> velocity;
        // Projection of velocities onto the forces [noi]
        std::vector<scalar> projection;
        // |force|^2
        std::vector<scalar> force_norm2;
	};
}

#endif