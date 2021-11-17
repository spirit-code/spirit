#pragma once
#ifndef SPIRIT_CORE_ENGINE_METHOD_HPP
#define SPIRIT_CORE_ENGINE_METHOD_HPP

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

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
    Method( std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain );

    // `Iterate` is supposed to iteratively solve a problem
    virtual void Iterate();

    // ------------------------------------------------------

    // Calculate a smooth but current IPS value
    virtual scalar getIterationsPerSecond() final;

    // Get the number of iterations passed
    virtual int getNIterations() final;

    // The amount of simulated time passed by the simulation
    virtual double get_simulated_time();

    // Get the number of milliseconds since the Method started iterating
    virtual int getWallTime() final;

    // NOTE: This is a bad convergence criterion and is therefore currently being phased out
    // Maximum of the absolutes of all components of the force - needs to be updated at each calculation
    virtual scalar getForceMaxAbsComponent() final;

    // Maximum of the absolutes of all components of the force for all images the method uses
    // The default is that this returns simply {getForceMaxAbsComponent()}
    virtual std::vector<scalar> getForceMaxAbsComponent_All();

    // Maximum norm of the torque - needs to be updated at each calculation
    virtual scalar getTorqueMaxNorm() final;

    // Maximum of the norm of the torque for all images the method uses
    virtual std::vector<scalar> getTorqueMaxNorm_All();

    // ------------------------------------------------------

    // Method name as string
    virtual std::string Name();

    // Solver name as string
    virtual std::string SolverName();
    virtual std::string SolverFullName();

    // ------------------------------------------------------

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
    virtual void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false );

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

    // Check wether to continue iterating - stop file, convergence etc.
    virtual bool ContinueIterating();

    //////////// Final implementations
    // Check if walltime ran out
    virtual bool Walltime_Expired( duration<scalar> dt_seconds ) final;
    // Check if a stop file is present -> Stop the iterations
    virtual bool StopFile_Present() final;

    std::chrono::time_point<std::chrono::system_clock> t_start, t_last;

    // Number of iterations that have been executed
    long iteration;
    // Number of steps (set of iterations between logs) that have been executed
    long step;

    // Timings and Iterations per Second
    scalar ips;
    std::deque<std::chrono::time_point<std::chrono::system_clock>> t_iterations;
    // The time at which this Solver's Iterate() was last called
    std::string starttime;

    //////////// Parameters //////////////////////////////////////////////////////
    // Number of iterations
    long n_iterations;
    // Number of iterations until log
    long n_iterations_log;
    // Number of times to save
    long n_log;

protected:
    //////////// Information for Logging and Save_Current ////////////////////////
    int idx_image;
    int idx_chain;

    // Number of images
    int noi;
    // Number of spins in an image
    int nos;

    // Method name as enum
    Utility::Log_Sender SenderName;
    // Maximum torque of all images
    scalar max_torque;
    // Maximum torque per image
    std::vector<scalar> max_torque_all;

    // Maximum force component of all images
    scalar force_max_abs_component;
    // Maximum force component per image
    std::vector<scalar> force_max_abs_component_all;

    // History of relevant quantities
    std::map<std::string, std::vector<scalar>> history;

    //////////// General /////////////////////////////////////////////////////////

    // Systems the Solver will access
    std::vector<std::shared_ptr<Data::Spin_System>> systems;

    // Method Parameters
    std::shared_ptr<Data::Parameters_Method> parameters;

    // Precision for the conversion of scalar to string
    int print_precision;
};

} // namespace Engine

#endif