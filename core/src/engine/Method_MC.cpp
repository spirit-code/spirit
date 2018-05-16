#include <Spirit_Defines.h>
#include <engine/Method_MC.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <io/IO.hpp>
#include <utility/Logging.hpp>

#include <iostream>
#include <ctime>
#include <math.h>

namespace Constants = Utility::Constants;

namespace Engine
{
    Method_MC::Method_MC(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
        Method(system->mc_parameters, idx_img, idx_chain)
    {
        // Currently we only support a single image being iterated at once:
        this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
        this->SenderName = Utility::Log_Sender::MC;

        this->noi = this->systems.size();
        this->nos = this->systems[0]->nos;

        this->xi = vectorfield(this->nos, {0,0,0});

        // We assume it is not converged before the first iteration
        // this->force_max_abs_component = system->mc_parameters->force_convergence + 1.0;

        // History
        this->history = std::map<std::string, std::vector<scalar>>{
            {"max_torque_component", {this->force_max_abs_component}},
            {"E", {this->force_max_abs_component}},
            {"M_z", {this->force_max_abs_component}} };

        this->parameters_mc = system->mc_parameters;

        // Starting cone angle
        this->cos_cone_angle = 0.1;
        this->n_rejected = 0;
        this->acceptance_ratio_current = this->parameters_mc->acceptance_ratio_target;
    }

    // Simple metropolis step for a single spin
    void Method_MC::Metropolis(const vectorfield & spins_old, const vectorfield & spins_displaced,
                               vectorfield & spins_new, int & n_rejected, scalar Temperature, scalar radius)
    {
        int nos = spins_new.size();
        auto distribution = std::uniform_real_distribution<scalar>(0, 1);
        auto distribution_idx = std::uniform_int_distribution<>(0, nos);
        scalar kB_T = Constants::k_B * Temperature;

        // One Metropolis step for each spin
        for (int irnd=0; irnd < nos; ++irnd)
        {
            int ispin = distribution_idx(this->parameters_mc->prng);

            // Displace the spin
            spins_new[ispin] = spins_displaced[ispin];

            // Energy difference of configurations with and without displacement
            scalar Eold = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_old);
            scalar Enew = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_new);

            scalar Ediff = Enew-Eold;

            // Metropolis criterion: reject the step if energy rose
            if (Ediff > 1e-14)
            {
                if (Temperature < 1e-12)
                {
                    // Restore the spin
                    spins_new[ispin] = spins_old[ispin];
                    // Counter for the number of rejections
                    ++n_rejected;
                }
                else
                {
                    // Exponential factor
                    scalar exp_ediff    = std::exp( -Ediff/kB_T );
                    // Metropolis random number
                    scalar x_metropolis = distribution(this->parameters_mc->prng);

                    // Only reject if random number is larger than exponential
                    if (exp_ediff < x_metropolis)
                    {
                        // Restore the spin
                        spins_new[ispin] = spins_old[ispin];
                        // Counter for the number of rejections
                        ++n_rejected;
                    }
                }
            }
        }
    }

    // This implementation is mostly serial as parallelization is nontrivial
    //      if the range of neighbours for each atom is not pre-defined.
    void Method_MC::Iteration()
    {
        int nos = this->systems[0]->spins->size();

        scalar diff = 1e-3;

        // Cone angle feedback algorithm
        this->acceptance_ratio_current = 1 - (scalar)this->n_rejected / (scalar)nos;
        if( (this->acceptance_ratio_current < this->parameters_mc->acceptance_ratio_target) && (this->cos_cone_angle > diff) )
        {
            this->cos_cone_angle -= diff;
        }
        if( (this->acceptance_ratio_current > this->parameters_mc->acceptance_ratio_target) && (this->cos_cone_angle < 1-diff) )
        {
            this->cos_cone_angle += diff;
        }

        // Temporaries
        auto& spins_old       = *this->systems[0]->spins;
        auto  spins_displaced = vectorfield(nos);
        auto  spins_new       = spins_old;

        // Generate randomly displaced spin configuration according to cone radius
        Vectormath::get_random_vectorfield_unitsphere(this->parameters_mc->prng, spins_displaced);
        Vectormath::scale(spins_displaced, cos_cone_angle);
        Vectormath::add_c_a(1, spins_old, spins_displaced);
        Vectormath::normalize_vectors(spins_displaced);

        // One Metropolis step
        this->n_rejected = 0;
        Metropolis(spins_old, spins_displaced, spins_new, this->n_rejected, this->parameters_mc->temperature, this->cos_cone_angle);
        Vectormath::set_c_a(1, spins_new, spins_old);
    }

    void Method_MC::Hook_Pre_Iteration()
    {
    }

    void Method_MC::Hook_Post_Iteration()
    {
    }

    void Method_MC::Initialize()
    {
    }

    void Method_MC::Finalize()
    {
        this->systems[0]->iteration_allowed = false;
    }

    void Method_MC::Message_Start()
    {
        using namespace Utility;

        //---- Log messages
        Log.SendBlock(Log_Level::All, this->SenderName,
        {
            fmt::format("------------  Started  {} Calculation  ------------", this->Name()),
            fmt::format("    Going to iterate {} steps", this->n_log),
            fmt::format("                with {} iterations per step", this->n_iterations_log),
            fmt::format("   Target acceptance {}", this->parameters_mc->acceptance_ratio_target),
            "-----------------------------------------------------"
        }, this->idx_image, this->idx_chain);
    }

    void Method_MC::Message_Step()
    {
        using namespace Utility;

        // Update time of current step
        auto t_current = system_clock::now();

        // Update the system's energy
        this->systems[0]->UpdateEnergy();

        // Send log message
        Log.SendBlock(Log_Level::All, this->SenderName,
        {
            fmt::format("----- {} Calculation: {}", this->Name(), Timing::DateTimePassed(t_current - this->t_start)),
            fmt::format("    Step                      {} / {} (step size {})", this->step, this->n_log, this->n_iterations_log),
            fmt::format("    Iteration                 {} / {}", this->iteration, this->n_iterations),
            fmt::format("    Time since last step:     {}", Timing::DateTimePassed(t_current - this->t_last)),
            fmt::format("    Iterations / sec:         {}", this->n_iterations_log / Timing::SecondsPassed(t_current - this->t_last)),
            fmt::format("    Current acceptance ratio: {} (target {})", this->acceptance_ratio_current, this->parameters_mc->acceptance_ratio_target),
            fmt::format("    Current cone angle:       {}", this->cos_cone_angle),
            fmt::format("    Total energy:             {:20.10f}", this->systems[0]->E)
        }, this->idx_image, this->idx_chain);

        // Update time of last step
        this->t_last = t_current;
    }

    void Method_MC::Message_End()
    {
        using namespace Utility;

        //---- End timings
        auto t_end = system_clock::now();

        //---- Termination reason
        std::string reason = "";
        if (this->StopFile_Present())
            reason = "A STOP file has been found";
        else if (this->Walltime_Expired(t_end - this->t_start))
            reason = "The maximum walltime has been reached";

        // Update the system's energy
        this->systems[0]->UpdateEnergy();

        //---- Log messages
        std::vector<std::string> block;
        block.push_back(fmt::format("------------ Terminated {} Calculation ------------", this->Name()));
        if (reason.length() > 0)
            block.push_back(fmt::format("----- Reason:   {}", reason));
        block.push_back(fmt::format("----- Duration:       {}", Timing::DateTimePassed(t_end - this->t_start)));
        block.push_back(fmt::format("    Step              {} / {}", step, n_log));
        block.push_back(fmt::format("    Iteration         {} / {}", this->iteration, n_iterations));
        block.push_back(fmt::format("    Iterations / sec: {}", this->iteration / Timing::SecondsPassed(t_end - this->t_start)));
        block.push_back(fmt::format("    Acceptance ratio: {} (target {})", this->acceptance_ratio_current, this->parameters_mc->acceptance_ratio_target));
        block.push_back(fmt::format("    Cone angle:       {}", this->cos_cone_angle));
        block.push_back(fmt::format("    Total energy:     {:20.10f}", this->systems[0]->E));
        block.push_back("-----------------------------------------------------");
        Log.SendBlock(Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain);
    }


    void Method_MC::Save_Current(std::string starttime, int iteration, bool initial, bool final)
    {
    }

    // Method name as string
    std::string Method_MC::Name() { return "MC"; }
}