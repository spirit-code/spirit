#pragma once
#ifndef Method_Solver_H
#define Method_Solver_H

#include "Spirit_Defines.h"
#include <data/Parameters_Method.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method.hpp>
#include <engine/Method.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Timing.hpp>
#include <utility/Logging.hpp>
#include <utility/Constants.hpp>

#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <iomanip>

#include <fmt/format.h>

namespace Engine
{
    enum class Solver
    {
        None,
        SIB,
        Heun,
        Depondt,
        NCG,
        BFGS,
        VP
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
        Method_Solver(std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain) :
            Method(parameters, idx_img, idx_chain)
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

    protected:

        // Calculate Forces onto Systems
        //      This is currently overridden by methods to specify how the forces on a set of configurations should be
        //      calculated. This function is used in `the Solver_...` functions.
        // TODO: maybe rename to separate from deterministic and stochastic force functions
        virtual void Calculate_Force(const std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<vectorfield> & forces)
        {

        }

        // Calculate virtual Forces onto Systems (can be precession and damping forces, correctly scaled)
        // Calculate the effective force on a configuration. It is a combination of
        //      precession and damping terms for the Hamiltonian, spin currents and
        //      temperature. This function is used in `the Solver_...` functions.
        // Default implementation: direct minimization
        virtual void Calculate_Force_Virtual(const std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<vectorfield> & forces, std::vector<vectorfield> & forces_virtual)
        {
            using namespace Utility;

            // Calculate the cross product with the spin configuration to get direct minimization
            for (unsigned int i=0; i<configurations.size(); ++i)
            {
                auto& image = *configurations[i];
                auto& force = forces[i];
                auto& force_virtual = forces_virtual[i];
                auto& parameters = *this->systems[i]->llg_parameters;

                // dt = time_step [ps] * gyromagnetic ratio / mu_B / (1+damping^2) <- not implemented
                scalar dtg = parameters.dt * Constants::gamma / Constants::mu_B;
                Vectormath::set_c_cross(0.5 * dtg, image, force, force_virtual);

                // Apply Pinning
                #ifdef SPIRIT_ENABLE_PINNING
                    Vectormath::set_c_a(1, force_virtual, force_virtual, parameters.pinning->mask_unpinned);
                #endif // SPIRIT_ENABLE_PINNING
            }
        }


        // Calculate maximum of absolute values of force components for a spin configuration
        virtual scalar Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force) final;

        // ...
        // virtual bool Iterations_Allowed() override;
        // Check if the forces are converged
        virtual bool Converged() override;

        // Iteration represents one iteration of a certain Solver
        virtual void Iteration() override;

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

        //////////// NCG ////////////////////////////////////////////////////////////
        // Check if the Newton-Raphson has converged
        bool NR_converged();
        
        int jmax;     // max iterations for Newton-Raphson loop
        int n;        // number of iteration after which the nCG will restart
        
        scalar tolerance_nCG, tolerance_NR;   // tolerances for solver and Newton-Raphson
        scalar epsilon_nCG, epsilon_NR;   // Newton-Raphson and solver tolerance squared
        
        bool restart_nCG, continue_NR;  // conditions for restarting nCG or continuing Newton-Raphson 
        
        // Step sizes
        std::vector<scalarfield> alpha, beta;
        
        // TODO: right type might be std::vector<scalar> and NOT std::vector<scalarfield>
        // Delta scalarfields
        std::vector<scalarfield> delta_0, delta_new, delta_old, delta_d;
        
        // Residual and new configuration states
        std::vector<vectorfield> residual, direction;

        // buffer variables for checking convergence for solver and Newton-Raphson
        std::vector<scalarfield> r_dot_d, dda2;

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
    };


    // Return the maximum of absolute values of force components for an image
    template<Solver solver>
    scalar Method_Solver<solver>::Force_on_Image_MaxAbsComponent(const vectorfield & image, vectorfield & force)
    {
        // Take out component in direction of v2
        Manifoldmath::project_tangential(force, image);

        // We want the Maximum of Absolute Values of all force components on all images
        return Vectormath::max_abs_component(force);
    }

    template<Solver solver>
    bool Method_Solver<solver>::Converged()
    {
        bool converged = false;
        if ( this->force_max_abs_component < this->parameters->force_convergence ) converged = true;
        return converged;
    }

    // Default implementation: do nothing
    template<Solver solver>
    void Method_Solver<solver>::Initialize()
    {
    };

    // Default implementation: do nothing
    template<Solver solver>
    void Method_Solver<solver>::Finalize()
    {
    };

    // Default implementation: do nothing
    template<Solver solver>
    void Method_Solver<solver>::Iteration()
    {
    };



    template<Solver solver>
    void Method_Solver<solver>::Message_Start()
    {
        using namespace Utility;

        //---- Log messages
        Log.SendBlock(Log_Level::All, this->SenderName,
            {
                "------------  Started  " + this->Name() + " Calculation  ------------",
                "    Going to iterate " + std::to_string(this->n_log) + " steps",
                "                with " + std::to_string(this->n_iterations_log) + " iterations per step",
                "    Force convergence parameter: " + fmt::format("{:." + std::to_string(this->print_precision) + "f}", this->parameters->force_convergence),
                "    Maximum force component:     " + fmt::format("{:." + std::to_string(this->print_precision) + "f}", this->force_max_abs_component),
                "    Solver: " + this->SolverFullName(),
                "-----------------------------------------------------"
            }, this->idx_image, this->idx_chain);
    }

    template<Solver solver>
    void Method_Solver<solver>::Message_Step()
    {
        using namespace Utility;
        
        // Update time of current step
        auto t_current = system_clock::now();

        // Send log message
        Log.SendBlock(Log_Level::All, this->SenderName,
            {
                "----- " + this->Name() + " Calculation (" + this->SolverName() + " Solver): " + Timing::DateTimePassed(t_current - this->t_start),
                "    Step                         " + std::to_string(step) + " / " + std::to_string(n_log),
                "    Iteration                    " + std::to_string( this->iteration) + " / " + std::to_string(n_iterations),
                "    Time since last step:        " + Timing::DateTimePassed(t_current - this->t_last),
                "    Iterations / sec:            " + std::to_string(this->n_iterations_log / Timing::SecondsPassed(t_current - this->t_last)),
                "    Force convergence parameter: " + fmt::format("{:." + std::to_string(this->print_precision) + "f}", this->parameters->force_convergence),
                "    Maximum force component:     " + fmt::format("{:." + std::to_string(this->print_precision) + "f}", this->force_max_abs_component)
            }, this->idx_image, this->idx_chain);

        // Update time of last step
        this->t_last = t_current;
    }

    template<Solver solver>
    void Method_Solver<solver>::Message_End()
    {
        using namespace Utility;
        
        //---- End timings
        auto t_end = system_clock::now();

        //---- Termination reason
        std::string reason = "";
        if (this->StopFile_Present())
            reason = "A STOP file has been found";
        else if (this->Converged())
            reason = "The force converged";
        else if (this->Walltime_Expired(t_end - this->t_start))
            reason = "The maximum walltime has been reached";

        //---- Log messages
        std::vector<std::string> block;
        block.push_back("------------ Terminated " + this->Name() + " Calculation ------------");
        if (reason.length() > 0)
            block.push_back("----- Reason:   " + reason);
        block.push_back("----- Duration:       " + Timing::DateTimePassed(t_end - this->t_start));
        block.push_back("    Step              " + std::to_string(step) + " / " + std::to_string(n_log));
        block.push_back("    Iteration         " + std::to_string( this->iteration) + " / " + std::to_string(n_iterations));
        block.push_back("    Iterations / sec: " + std::to_string( this->iteration / Timing::SecondsPassed(t_end - this->t_start)));
        block.push_back("    Force convergence parameter: " + fmt::format("{:."+std::to_string(this->print_precision)+"f}", this->parameters->force_convergence));
        block.push_back("    Maximum force component:     " + fmt::format("{:."+std::to_string(this->print_precision)+"f}", this->force_max_abs_component));
        block.push_back("    Solver: " + this->SolverFullName());
        block.push_back("-----------------------------------------------------");
        Log.SendBlock(Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain);
    }


    template <> inline
    std::string Method_Solver<Solver::None>::SolverName()
    {
        return "None";
    };

    template <> inline
    std::string Method_Solver<Solver::None>::SolverFullName()
    {
        return "None";
    };

    // Include headers which specialize the Solver functions
    #include <engine/Solver_SIB.hpp>
    #include <engine/Solver_VP.hpp>
    #include <engine/Solver_Heun.hpp>
    #include <engine/Solver_Depondt.hpp>
    #include <engine/Solver_NCG.hpp>
}

#endif