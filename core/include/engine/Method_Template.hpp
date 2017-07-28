#pragma once
#ifndef METHOD_TEMPLATE_H
#define METHOD_TEMPLATE_H

#include <Method.hpp>

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Timing.hpp>
#include <utility/Logging.hpp>
#include <utility/Constants.hpp>

#include <deque>
#include <fstream>
#include <map>

namespace Engine
{
    enum class Solver
    {
        SIB,
        Heun,
        Depondt,
        NCG,
        BFGS,
        VP
    };

    /*
        Base Class for Simulation/Calculation Methods.
        It is templated to allow a flexible choice of Solver to iterate the system.

    */
    template<Solver solver>
    class Method_Template : public Method
    {
    public:
    // 	// Constructor to be used in derived classes
    // 	Method_Template(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain);

        // Solver_Initialise contains the initialisations of arrays etc. for a certain solver
        void Solver_Init();
        // Solver_Step represents one iteration of a certain Solver
        void Solver_Step();


    // 	// Check if walltime ran out
    // 	virtual bool Walltime_Expired(duration<scalar> dt_seconds);

    // 	// Calculate Forces onto Systems
    // 	virtual void Calculate_Force(std::vector<std::shared_ptr<vectorfield>> configurations, std::vector<vectorfield> & forces);
    // 	// Maximum of the absolutes of all components of the force - needs to be updated at each calculation
    // 	scalar force_maxAbsComponent;
    // 	// Check if the Forces are converged
    // 	virtual bool Force_Converged();

    // 	// Check wether to continue iterating - stop file, convergence etc.
    // 	virtual bool ContinueIterating() final;

    // 	// A hook into the Optimizer before an Iteration
    // 	virtual void Hook_Pre_Iteration();
    // 	// A hook into the Optimizer after an Iteration
    // 	virtual void Hook_Post_Iteration();

    // Save the current Step's Data: spins and energy
    // virtual void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false) override;
    // // A hook into the Optimizer before an Iteration
    // virtual void Hook_Pre_Iteration() override;
    // // A hook into the Optimizer after an Iteration
    // virtual void Hook_Post_Iteration() override;

    // // Sets iteration_allowed to false for the corresponding method
    // virtual void Finalize() override;

    // 	// Lock systems in order to prevent otherwise access
    // 	virtual void Lock();
    // 	// Unlock systems to re-enable access
    // 	virtual void Unlock();

    // 	// Method name as string
    // 	virtual std::string Name();
    // 	// Method name as enum
    // 	Utility::Log_Sender SenderName;
        
    // 	// // Systems the Optimizer will access
    // 	std::vector<std::shared_ptr<Data::Spin_System>> systems;
    // 	// Method Parameters
    // 	std::shared_ptr<Data::Parameters_Method> parameters;

    // 	// Information for Logging and Save_Current
    // 	int idx_image;
    // 	int idx_chain;
        
    // protected:
    // 	// Calculate force_maxAbsComponent for a spin configuration
    // 	virtual scalar Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force) final;
    // 	// Check if iterations_allowed
    // 	virtual bool Iterations_Allowed();
    // 	// History of relevant quantities
    // 	std::map<std::string, std::vector<scalar>> history;

    protected:
        // Temporary Spins arrays
        std::vector<std::shared_ptr<vectorfield>> spins_temp;

        // Virtual Forces used in the Steps
        std::vector<vectorfield> virtualforce;

        // Random vector array
        vectorfield xi;
        // Some variable
        scalar epsilon;

        // Temporary Spins arrays
        std::vector<std::shared_ptr<vectorfield>> spins_predictor;
        vectorfield temp1, temp2;
        
        // pointer to spin system
        std::shared_ptr<Data::Spin_System> s;
        
        // method time step
        scalar dt;


        scalar dtg;
        
        
        // Virtual force
        std::vector<vectorfield> virtualforce_predictor;
        std::vector<vectorfield> rotationaxis;

        std::vector<scalarfield> virtualforce_norm;
        
        // preccession angle
        scalarfield angle;

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

    #include <Solver_SIB.hpp>
    #include <Solver_VP.hpp>
    #include <Solver_Heun.hpp>
    #include <Solver_Depondt.hpp>
    #include <Solver_NCG.hpp>
}

#endif