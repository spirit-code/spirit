#pragma once
#ifndef Method_CORE_ENGINE_Solver_HPP
#define Method_CORE_ENGINE_Solver_HPP

#include "Spirit_Defines.h"
#include <Spirit/Simulation.h>
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Method.hpp>
#include <engine/Solver_Kernels.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <deque>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <fmt/format.h>

namespace Engine
{
enum class Solver
{
    None        = -1,
    SIB         = Solver_SIB,
    Heun        = Solver_Heun,
    Depondt     = Solver_Depondt,
    RungeKutta4 = Solver_RungeKutta4,
    LBFGS_OSO   = Solver_LBFGS_OSO,
    LBFGS_Atlas = Solver_LBFGS_Atlas,
    VP          = Solver_VP,
    VP_OSO      = Solver_VP_OSO
};

/*
    Base Class for Solver-based Simulation/Calculation Methods.
    It is templated to allow a flexible choice of Solver to iterate the systems.
*/
template<Solver solver>
class Method_Solver : public Method
{
public:
    // Constructor to be used in derived classes
    Method_Solver( std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain )
            : Method( parameters, idx_img, idx_chain )
    {
    }

    // // `Iterate` uses the `Solver_Iteration` function to evolve given systems according to the
    // // `Calculate_Force` implementation of the Method-Subclass.
    // //      It iterates until: the maximum number of iterations is reached or the maximum
    // //      walltime is reaches or the force has converged or a file called `STOP` is found
    // //      or the calculation is stopped externally (via the API).
    // virtual void Iterate() override;

    // Solver name as string
    virtual std::string SolverName() override;
    virtual std::string SolverFullName() override;

    // Iteration represents one iteration of a certain Solver
    virtual void Iteration() override;

protected:
    // Prepare random numbers for thermal fields, if needed
    virtual void Prepare_Thermal_Field() {}

    // Calculate Forces onto Systems
    //      This is currently overridden by methods to specify how the forces on a set of configurations should be
    //      calculated. This function is used in `the Solver_...` functions.
    // TODO: maybe rename to separate from deterministic and stochastic force functions
    virtual void Calculate_Force(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             "Tried to use Method_Solver::Calculate_Force() of the Method_Solver class!", this->idx_image,
             this->idx_chain );
    }

    // Calculate virtual Forces onto Systems (can be precession and damping forces, correctly scaled)
    // Calculate the effective force on a configuration. It is a combination of
    //      precession and damping terms for the Hamiltonian, spin currents and
    //      temperature. This function is used in `the Solver_...` functions.
    // Default implementation: direct minimization
    virtual void Calculate_Force_Virtual(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces,
        std::vector<vectorfield> & forces_virtual )
    {
        // Not Implemented!
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             "Tried to use Method_Solver::Calculate_Force_Virtual() of the Method_Solver class!", this->idx_image,
             this->idx_chain );
    }

    // Calculate maximum of absolute values of force components for a spin configuration
    virtual scalar Force_on_Image_MaxAbsComponent( const vectorfield & image, vectorfield & force ) final;

    // Calculate maximum torque for a spin configuration
    virtual scalar MaxTorque_on_Image( const vectorfield & image, vectorfield & force ) final;

    // ...
    // virtual bool Iterations_Allowed() override;
    // Check if the forces are converged
    virtual bool Converged();

    // Check if any stop criteria were encountered
    bool ContinueIterating() override
    {
        return Method::ContinueIterating() && !this->Converged();
    }

    // Initialise contains the initialisations of arrays etc. for a certain solver
    virtual void Initialize() override;
    virtual void Finalize() override;

    // Log message blocks
    virtual void Message_Start() override;
    virtual void Message_Step() override;
    virtual void Message_End() override;

    //////////// DEPONDT ////////////////////////////////////////////////////////////
    // Temporaries for virtual forces
    std::vector<vectorfield> rotationaxis;
    std::vector<scalarfield> forces_virtual_norm;
    // Preccession angle
    scalarfield angle;

    //////////// LBFGS ////////////////////////////////////////////////////////////

    // General
    int n_lbfgs_memory;
    int local_iter;
    scalar maxmove;
    scalarfield rho;
    scalarfield alpha;

    // Atlas coords
    std::vector<field<vector2field>> atlas_updates;
    std::vector<field<vector2field>> grad_atlas_updates;
    std::vector<scalarfield> atlas_coords3;
    std::vector<vector2field> atlas_directions;
    std::vector<vector2field> atlas_residuals;
    std::vector<vector2field> atlas_residuals_last;
    std::vector<vector2field> atlas_q_vec;

    // OSO
    std::vector<field<vectorfield>> delta_a;
    std::vector<field<vectorfield>> delta_grad;
    std::vector<vectorfield> searchdir;
    std::vector<vectorfield> grad;
    std::vector<vectorfield> grad_pr;
    std::vector<vectorfield> q_vec;

    // buffer variables for checking convergence for solver and Newton-Raphson
    // std::vector<scalarfield> r_dot_d, dda2;

    //////////// VP ///////////////////////////////////////////////////////////////
    // "Mass of our particle" which we accelerate
    scalar m = 1.0;

    // Force in previous step [noi][nos]
    std::vector<vectorfield> forces_previous;
    // Velocity in previous step [noi][nos]
    std::vector<vectorfield> velocities_previous;
    // Velocity used in the Steps [noi][nos]
    std::vector<vectorfield> velocities;
    // Projection of velocities onto the forces [noi]
    std::vector<scalar> projection;
    // |force|^2
    std::vector<scalar> force_norm2;

    // Temporary Spins arrays
    vectorfield temp1, temp2;

    // Actual Forces on the configurations
    std::vector<vectorfield> forces;
    std::vector<vectorfield> forces_predictor;
    // Virtual Forces used in the Steps
    std::vector<vectorfield> forces_virtual;
    std::vector<vectorfield> forces_virtual_predictor;

    // RK 4
    std::vector<std::shared_ptr<vectorfield>> configurations_k1;
    std::vector<std::shared_ptr<vectorfield>> configurations_k2;
    std::vector<std::shared_ptr<vectorfield>> configurations_k3;
    std::vector<std::shared_ptr<vectorfield>> configurations_k4;

    // Random vector array
    vectorfield xi;

    // Pointers to Configurations (for Solver methods)
    std::vector<std::shared_ptr<vectorfield>> configurations;
    std::vector<std::shared_ptr<vectorfield>> configurations_predictor;
    std::vector<std::shared_ptr<vectorfield>> configurations_temp;
};

// Return the maximum of absolute values of force components for an image
template<Solver solver>
scalar Method_Solver<solver>::Force_on_Image_MaxAbsComponent( const vectorfield & image, vectorfield & force )
{
    // Take out component in direction of v2
    Manifoldmath::project_tangential( force, image );

    // We want the Maximum of Absolute Values of all force components on all images
    return Vectormath::max_abs_component( force );
}

// Return the maximum norm of the torque for an image
template<Solver solver>
scalar Method_Solver<solver>::MaxTorque_on_Image( const vectorfield & image, vectorfield & force )
{
    // Take out component in direction of v2
    Manifoldmath::project_tangential( force, image );
    return Vectormath::max_norm( force );
}

template<Solver solver>
bool Method_Solver<solver>::Converged()
{
    bool converged = false;
    if( this->max_torque < this->parameters->force_convergence )
        converged = true;
    return converged;
}

// Default implementation: do nothing
template<Solver solver>
void Method_Solver<solver>::Initialize(){};

// Default implementation: do nothing
template<Solver solver>
void Method_Solver<solver>::Finalize(){};

// Default implementation: do nothing
template<Solver solver>
void Method_Solver<solver>::Iteration(){};

template<Solver solver>
void Method_Solver<solver>::Message_Start()
{
    using namespace Utility;

    //---- Log messages
    std::vector<std::string> block;
    block.push_back( fmt::format( "------------  Started  {} Calculation  ------------", this->Name() ) );
    block.push_back( fmt::format( "    Going to iterate {} step(s)", this->n_log ) );
    block.push_back( fmt::format( "                with {} iterations per step", this->n_iterations_log ) );
    block.push_back( fmt::format(
        "    Force convergence parameter: {:." + fmt::format( "{}", this->print_precision ) + "f}",
        this->parameters->force_convergence ) );
    block.push_back( fmt::format(
        "    Maximum torque:              {:." + fmt::format( "{}", this->print_precision ) + "f}",
        this->max_torque ) );
    block.push_back( fmt::format( "    Solver: {}", this->SolverFullName() ) );
    if( this->Name() == "GNEB" )
    {
        scalar length = Manifoldmath::dist_geodesic( *this->configurations[0], *this->configurations[this->noi - 1] );
        block.push_back( fmt::format( "    Total path length: {}", length ) );
    }
    block.push_back( "-----------------------------------------------------" );
    Log.SendBlock( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

template<Solver solver>
void Method_Solver<solver>::Message_Step()
{
    using namespace Utility;

    std::string percentage = fmt::format( "{:.2f}%:", 100 * double( this->iteration ) / double( this->n_iterations ) );
    bool llg_dynamics
        = this->Name() == "LLG"
          && !(
              this->systems[0]->llg_parameters->direct_minimization || solver == Solver::VP || solver == Solver::VP_OSO
              || solver == Solver::LBFGS_OSO || solver == Solver::LBFGS_Atlas );

    // Update time of current step
    auto t_current = system_clock::now();

    // Send log message
    std::vector<std::string> block;
    block.push_back( fmt::format(
        "----- {} Calculation ({} Solver): {}", this->Name(), this->SolverName(),
        Timing::DateTimePassed( t_current - this->t_start ) ) );
    block.push_back(
        fmt::format( "    Time since last step: {}", Timing::DateTimePassed( t_current - this->t_last ) ) );
    block.push_back(
        fmt::format( "    Completed {:>8}    {} / {} iterations", percentage, this->iteration, this->n_iterations ) );
    block.push_back( fmt::format(
        "    Iterations / sec:     {:.2f}",
        this->n_iterations_log / Timing::SecondsPassed( t_current - this->t_last ) ) );
    if( llg_dynamics )
        block.push_back( fmt::format( "    Simulated time:       {} ps", this->get_simulated_time() ) );
    if( this->Name() == "GNEB" )
    {
        scalar length = Manifoldmath::dist_geodesic( *this->configurations[0], *this->configurations[this->noi - 1] );
        block.push_back( fmt::format( "    Total path length:    {}", length ) );
    }
    block.push_back( fmt::format(
        "    Force convergence parameter: {:." + fmt::format( "{}", this->print_precision ) + "f}",
        this->parameters->force_convergence ) );
    block.push_back( fmt::format(
        "    Maximum torque:              {:." + fmt::format( "{}", this->print_precision ) + "f}",
        this->max_torque ) );
    Log.SendBlock( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );

    // Update time of last step
    this->t_last = t_current;
}

template<Solver solver>
void Method_Solver<solver>::Message_End()
{
    using namespace Utility;

    std::string percentage = fmt::format( "{:.2f}%:", 100 * double( this->iteration ) / double( this->n_iterations ) );
    bool llg_dynamics
        = this->Name() == "LLG"
          && !(
              this->systems[0]->llg_parameters->direct_minimization || solver == Solver::VP || solver == Solver::VP_OSO
              || solver == Solver::LBFGS_OSO || solver == Solver::LBFGS_Atlas );

    //---- End timings
    auto t_end = system_clock::now();

    //---- Termination reason
    std::string reason = "";
    if( this->StopFile_Present() )
        reason = "A STOP file has been found";
    else if( this->Converged() )
        reason = "The force converged";
    else if( this->Walltime_Expired( t_end - this->t_start ) )
        reason = "The maximum walltime has been reached";

    //---- Log messages
    std::vector<std::string> block;
    block.push_back( fmt::format( "------------ Terminated {} Calculation ------------", this->Name() ) );
    if( reason.length() > 0 )
        block.push_back( fmt::format( "------- Reason: {}", reason ) );
    block.push_back( fmt::format( "    Total duration:    {}", Timing::DateTimePassed( t_end - this->t_start ) ) );
    block.push_back(
        fmt::format( "    Completed {:>8} {} / {} iterations", percentage, this->iteration, this->n_iterations ) );
    block.push_back( fmt::format(
        "    Iterations / sec:  {:.2f}", this->iteration / Timing::SecondsPassed( t_end - this->t_start ) ) );
    if( llg_dynamics )
        block.push_back( fmt::format( "    Simulated time:    {} ps", this->get_simulated_time() ) );
    if( this->Name() == "GNEB" )
    {
        scalar length = Manifoldmath::dist_geodesic( *this->configurations[0], *this->configurations[this->noi - 1] );
        block.push_back( fmt::format( "    Total path length: {}", length ) );
    }
    block.push_back( fmt::format(
        "    Force convergence parameter: {:." + fmt::format( "{}", this->print_precision ) + "f}",
        this->parameters->force_convergence ) );
    block.push_back( fmt::format(
        "    Maximum torque:              {:." + fmt::format( "{}", this->print_precision ) + "f}",
        this->max_torque ) );
    block.push_back( fmt::format( "    Solver: {}", this->SolverFullName() ) );
    block.push_back( "-----------------------------------------------------" );
    Log.SendBlock( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

template<>
inline std::string Method_Solver<Solver::None>::SolverName()
{
    return "None";
};

template<>
inline std::string Method_Solver<Solver::None>::SolverFullName()
{
    return "None";
};

// Include headers which specialize the Solver functions
#include <engine/Solver_Depondt.hpp>
#include <engine/Solver_Heun.hpp>
#include <engine/Solver_LBFGS_Atlas.hpp>
#include <engine/Solver_LBFGS_OSO.hpp>
#include <engine/Solver_RK4.hpp>
#include <engine/Solver_SIB.hpp>
#include <engine/Solver_VP.hpp>
#include <engine/Solver_VP_OSO.hpp>
} // namespace Engine

#endif