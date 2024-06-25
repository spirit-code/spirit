#pragma once
#ifndef Method_CORE_ENGINE_Solver_HPP
#define Method_CORE_ENGINE_Solver_HPP

#include <Spirit/Simulation.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Solver_Kernels.hpp>
#include <engine/Vectormath.hpp>
#include <engine/common/Method_Solver.hpp>
#include <engine/spin/Method.hpp>
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

namespace Spin
{

enum struct Solver
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

constexpr auto common_solver( Spin::Solver solver ) -> Common::Solver
{
    switch( solver )
    {
        case Solver::None: return Common::Solver::None;
        case Solver::SIB: return Common::Solver::SIB;
        case Solver::Heun: return Common::Solver::Heun;
        case Solver::Depondt: return Common::Solver::Depondt;
        case Solver::RungeKutta4: return Common::Solver::RungeKutta4;
        case Solver::LBFGS_OSO: return Common::Solver::LBFGS_OSO;
        case Solver::LBFGS_Atlas: return Common::Solver::LBFGS_Atlas;
        case Solver::VP: return Common::Solver::VP;
        case Solver::VP_OSO: return Common::Solver::VP_OSO;
    }
}

constexpr auto name( Spin::Solver solver ) -> std::string_view
{
    switch( solver )
    {
        case Solver::None: return "None";
        case Solver::SIB: return "SIB";
        case Solver::Heun: return "Heun";
        case Solver::Depondt: return "Depondt";
        case Solver::RungeKutta4: return "RK4";
        case Solver::LBFGS_OSO: return "LBFGS_OSO";
        case Solver::LBFGS_Atlas: return "LBFGS_Atlas";
        case Solver::VP: return "VP";
        case Solver::VP_OSO: return "VP_OSO";
        default: return "Unknown";
    }
}

constexpr auto full_name( Spin::Solver solver ) -> std::string_view
{
    switch( solver )
    {
        case Solver::None: return "None";
        case Solver::SIB: return "Semi-implicit B";
        case Solver::Heun: return "Heun";
        case Solver::Depondt: return "Depondt";
        case Solver::RungeKutta4: return "Runge Kutta (4th order)";
        case Solver::LBFGS_OSO: return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using exponential transforms";
        case Solver::LBFGS_Atlas: return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using stereographic atlas";
        case Solver::VP: return "Velocity Projection";
        case Solver::VP_OSO: return "Velocity Projection using exponential transforms";
        default: return "Unknown";
    }
}

class SolverMethods : public Method
{
protected:
    using Method::Method;

    virtual void Prepare_Thermal_Field() = 0;
    virtual void Calculate_Force(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces ) = 0;
    virtual void Calculate_Force_Virtual(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces,
        std::vector<vectorfield> & forces_virtual ) = 0;
    // Actual Forces on the configurations
    std::vector<vectorfield> forces;
    // Virtual Forces used in the Steps
    std::vector<vectorfield> forces_virtual;
    // Pointers to Configurations (for Solver methods)
    std::vector<std::shared_ptr<vectorfield>> configurations;
};

// default implementation (to be overwritten by class template specialization)
template<Solver solver>
class SolverData : public SolverMethods
{
protected:
    using SolverMethods::SolverMethods;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
};

/*
 * Base Class for Solver-based Simulation/Calculation Methods.
 * It is templated to allow a flexible choice of Solver to iterate the systems.
 */
template<Solver solver>
class Method_Solver : public SolverData<solver>
{
public:
    // Constructor to be used in derived classes
    Method_Solver( std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain )
            : SolverData<solver>( parameters, idx_img, idx_chain )
    {
    }

    virtual ~Method_Solver() = default;

    // // `Iterate` uses the `Solver_Iteration` function to evolve given systems according to the
    // // `Calculate_Force` implementation of the Method-Subclass.
    // //      It iterates until: the maximum number of iterations is reached or the maximum
    // //      walltime is reaches or the force has converged or a file called `STOP` is found
    // //      or the calculation is stopped externally (via the API).
    // virtual void Iterate() override;

    // Lock systems in order to prevent otherwise access
    void Lock() override;
    // Unlock systems to re-enable access
    void Unlock() override;
    // Check if iterations are allowed
    bool Iterations_Allowed() override;

    // Solver name as string
    std::string_view SolverName() override;
    std::string_view SolverFullName() override;

    // Iteration represents one iteration of a certain Solver
    void Iteration() override;

protected:
    // Prepare random numbers for thermal fields, if needed
    void Prepare_Thermal_Field() override {}

    /*
     * Calculate Forces onto Systems
     *   This is currently overridden by methods to specify how the forces on a set of configurations should be
     *   calculated. This function is used in `the Solver_...` functions.
     * TODO: maybe rename to separate from deterministic and stochastic force functions
     */
    void Calculate_Force(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces ) override
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             "Tried to use Method_Solver::Calculate_Force() of the Method_Solver class!", this->idx_image,
             this->idx_chain );
    }

    /*
     * Calculate virtual Forces onto Systems (can be precession and damping forces, correctly scaled)
     * Calculate the effective force on a configuration. It is a combination of
     *   precession and damping terms for the Hamiltonian, spin currents and
     *   temperature. This function is used in `the Solver_...` functions.
     * Default implementation: direct minimization
     */
    void Calculate_Force_Virtual(
        const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces,
        std::vector<vectorfield> & forces_virtual ) override
    {
        // Not Implemented!
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             "Tried to use Method_Solver::Calculate_Force_Virtual() of the Method_Solver class!", this->idx_image,
             this->idx_chain );
    }

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
    void Initialize() override;
    void Finalize() override;

    // Log message blocks
    void Message_Start() final;
    void Message_Step() final;
    void Message_End() final;

    // Log message Customization point for solver specific data
    virtual void Message_Block_Start( std::vector<std::string> & block );
    virtual void Message_Block_Step( std::vector<std::string> & block );
    virtual void Message_Block_End( std::vector<std::string> & block );

    std::vector<std::shared_ptr<system_t>> systems;

};

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

template<Solver solver>
void Method_Solver<solver>::Lock()
{
    std::for_each( systems.begin(), systems.end(), []( const std::shared_ptr<system_t> & system ) { system->Lock(); } );
};

template<Solver solver>
void Method_Solver<solver>::Unlock()
{
    std::for_each(
        systems.begin(), systems.end(), []( const std::shared_ptr<system_t> & system ) { system->Unlock(); } );
};

template<Solver solver>
bool Method_Solver<solver>::Iterations_Allowed()
{
    return this->systems[0]->iteration_allowed;
};

// Default implementation: do nothing
template<Solver solver>
void Method_Solver<solver>::Iteration(){};

template<Solver solver>
void Method_Solver<solver>::Message_Start()
{
    using namespace Utility;

    //---- Log messages
    std::vector<std::string> block;
    block.emplace_back( fmt::format( "------------  Started  {} Calculation  ------------", this->Name() ) );
    block.emplace_back( fmt::format( "    Going to iterate {} step(s)", this->n_log ) );
    block.emplace_back( fmt::format( "                with {} iterations per step", this->n_iterations_log ) );
    block.emplace_back( fmt::format(
        fmt::format( "    Force convergence parameter: {{:.{}f}}", Method::print_precision ),
        this->parameters->force_convergence ) );
    block.emplace_back(
        fmt::format( fmt::format( "    Maximum torque:              {{:.{}f}}", Method::print_precision ), this->max_torque ) );
    block.emplace_back( fmt::format( "    Solver: {}", this->SolverFullName() ) );
    // solver specific message
    this->Message_Block_Start( block );
    block.emplace_back( "-----------------------------------------------------" );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}
template<Solver solver>
void Method_Solver<solver>::Message_Block_Start( std::vector<std::string> & )
{
}

template<Solver solver>
void Method_Solver<solver>::Message_Block_Step( std::vector<std::string> & )
{
}

template<Solver solver>
void Method_Solver<solver>::Message_Block_End( std::vector<std::string> & )
{
}

template<Solver solver>
void Method_Solver<solver>::Message_Step()
{
    using namespace Utility;

    std::string percentage = fmt::format( "{:.2f}%:", 100 * double( this->iteration ) / double( this->n_iterations ) );

    // Update time of current step
    auto t_current = std::chrono::system_clock::now();

    // Send log message
    std::vector<std::string> block;
    block.emplace_back( fmt::format(
        "----- {} Calculation ({} Solver): {}", this->Name(), this->SolverName(),
        Timing::DateTimePassed( t_current - this->t_start ) ) );
    block.emplace_back(
        fmt::format( "    Time since last step: {}", Timing::DateTimePassed( t_current - this->t_last ) ) );
    block.emplace_back(
        fmt::format( "    Completed {:>8}    {} / {} iterations", percentage, this->iteration, this->n_iterations ) );
    block.emplace_back( fmt::format(
        "    Iterations / sec:     {:.2f}",
        this->n_iterations_log / Timing::SecondsPassed( t_current - this->t_last ) ) );
    // solver specific message
    this->Message_Block_Step( block );
    block.emplace_back( fmt::format(
        fmt::format( "    Force convergence parameter: {{:.{}f}}", Method::print_precision ),
        this->parameters->force_convergence ) );
    block.emplace_back(
        fmt::format( fmt::format( "    Maximum torque:              {{:.{}f}}", Method::print_precision ), this->max_torque ) );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );

    // Update time of last step
    this->t_last = t_current;
}

template<Solver solver>
void Method_Solver<solver>::Message_End()
{
    using namespace Utility;

    std::string percentage = fmt::format( "{:.2f}%:", 100 * double( this->iteration ) / double( this->n_iterations ) );

    //---- End timings
    auto t_end = std::chrono::system_clock::now();

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
    block.emplace_back( fmt::format( "------------ Terminated {} Calculation ------------", this->Name() ) );
    if( reason.length() > 0 )
        block.emplace_back( fmt::format( "------- Reason: {}", reason ) );
    block.emplace_back( fmt::format( "    Total duration:    {}", Timing::DateTimePassed( t_end - this->t_start ) ) );
    block.emplace_back(
        fmt::format( "    Completed {:>8} {} / {} iterations", percentage, this->iteration, this->n_iterations ) );
    block.emplace_back( fmt::format(
        "    Iterations / sec:  {:.2f}", this->iteration / Timing::SecondsPassed( t_end - this->t_start ) ) );
    // solver specific message
    this->Message_Block_End( block );
    block.emplace_back( fmt::format(
        fmt::format( "    Force convergence parameter: {{:.{}f}}", Method::print_precision ),
        this->parameters->force_convergence ) );
    block.emplace_back(
        fmt::format( fmt::format( "    Maximum torque:              {{:.{}f}}", Method::print_precision ), this->max_torque ) );
    block.emplace_back( fmt::format( "    Solver: {}", this->SolverFullName() ) );
    block.emplace_back( "-----------------------------------------------------" );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

template<Solver solver>
std::string_view Method_Solver<solver>::SolverName()
{
    return name( solver );
};

template<Solver solver>
std::string_view Method_Solver<solver>::SolverFullName()
{
    return full_name( solver );
};

} // namespace Spin

} // namespace Engine

// Include headers which specialize the Solver functions
#include <engine/spin/Solver_Depondt.hpp>
#include <engine/spin/Solver_Heun.hpp>
#include <engine/spin/Solver_LBFGS_Atlas.hpp>
#include <engine/spin/Solver_LBFGS_OSO.hpp>
#include <engine/spin/Solver_RK4.hpp>
#include <engine/spin/Solver_SIB.hpp>
#include <engine/spin/Solver_VP.hpp>
#include <engine/spin/Solver_VP_OSO.hpp>

#endif
