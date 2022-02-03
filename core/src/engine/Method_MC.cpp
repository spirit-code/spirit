#include <Spirit_Defines.h>
#include <engine/Method_MC.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <io/IO.hpp>
#include <utility/Logging.hpp>
#include <utility/Constants.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <ctime>
#include <math.h>
#include <curand_kernel.h>
#include <curand.h>
using namespace Utility;

namespace Engine
{
    Method_MC::Method_MC(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
        Method(system->mc_parameters, idx_img, idx_chain)
    {
        // Currently we only support a single image being iterated at once:
        this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
        this->SenderName = Log_Sender::MC;

        this->noi = this->systems.size();
        this->nos = this->systems[0]->geometry->nos;
        this->nos_nonvacant = this->systems[0]->geometry->nos_nonvacant;

        this->xi = vectorfield(this->nos, {0,0,0});

        // We assume it is not converged before the first iteration
        // this->max_torque = system->mc_parameters->force_convergence + 1.0;

        // History
        this->history = std::map<std::string, std::vector<scalar>>{
            {"max_torque", {this->max_torque}},
            {"E", {this->max_torque}},
            {"M_z", {this->max_torque}} };

        this->parameters_mc = system->mc_parameters;

        // Starting cone angle
        this->cone_angle = Constants::Pi * this->parameters_mc->metropolis_cone_angle / 180.0;
        this->n_rejected = 0;
        this->acceptance_ratio_current = this->parameters_mc->acceptance_ratio_target;
        Block_Decomposition();
        Setup_Curand();
    }

    // This implementation is mostly serial as parallelization is nontrivial
    //      if the range of neighbours for each atom is not pre-defined.
    void Method_MC::Iteration()
    {
        // Temporaries
        auto& spins_old       = *this->systems[0]->spins;
        auto  spins_new       = spins_old;

        // Generate randomly displaced spin configuration according to cone radius
        // Vectormath::get_random_vectorfield_unitsphere(this->parameters_mc->prng, random_unit_vectors);

        // TODO: add switch between Metropolis and heat bath
        // One Metropolis step
        Parallel_Metropolis(spins_old, spins_new);
        // Metropolis(spins_old, spins_new);
        Vectormath::set_c_a(1, spins_new, spins_old);
    }

    bool Method_MC::Metropolis_Spin_Trial(int ispin, const vectorfield & spins_old, vectorfield & spins_new, const scalar & rng1, const scalar & rng2, const scalar & rng3, const scalar & cos_cone_angle)
    {
        Matrix3 local_basis;
        const Vector3 e_z{0,0,1};
        const scalar kB_T = Constants::k_B * this->parameters_mc->temperature;

        // Calculate local basis for the spin
        if( std::abs(spins_old[ispin].z()) < 1-1e-10 )
        {
            local_basis.col(2) = spins_old[ispin];
            local_basis.col(0) = (local_basis.col(2).cross(e_z)).normalized();
            local_basis.col(1) = local_basis.col(2).cross(local_basis.col(0));
        } else
        {
            local_basis = Matrix3::Identity();
        }

        // Rotation angle between 0 and cone_angle degrees
        scalar costheta = 1 - (1 - cos_cone_angle) * rng1;

        scalar sintheta = std::sqrt(1 - costheta*costheta);

        // Random distribution of phi between 0 and 360 degrees
        scalar phi = 2*Constants::Pi * rng2;

        Vector3 local_spin_new{ sintheta * std::cos(phi),
                                sintheta * std::sin(phi),
                                costheta };

        // New spin orientation in regular basis
        spins_new[ispin] = local_basis * local_spin_new;

        // Energy difference of configurations with and without displacement
        scalar Eold  = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_old);
        scalar Enew  = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_new);
        scalar Ediff = Enew-Eold;

        // Metropolis criterion: reject the step if energy rose
        if( Ediff > 1e-14 )
        {
            if( this->parameters_mc->temperature < 1e-12 )
            {
                // Restore the spin
                spins_new[ispin] = spins_old[ispin];
                return false;
            }
            else
            {
                // Exponential factor
                scalar exp_ediff    = std::exp( -Ediff/kB_T );
                // Only reject if random number is larger than exponential
                if( exp_ediff < rng3 )
                {
                    // Restore the spin
                    spins_new[ispin] = spins_old[ispin];
                    // Counter for the number of rejections
                    return false;
                }
            }
        }
        return true;
    }

    // Simple metropolis step
    void Method_MC::Metropolis(const vectorfield & spins_old, vectorfield & spins_new)
    {
        auto distribution = std::uniform_real_distribution<scalar>(0, 1);
        auto distribution_idx = std::uniform_int_distribution<>(0, this->nos-1);
        scalar kB_T = Constants::k_B * this->parameters_mc->temperature;

        scalar diff = 0.01;

        // Cone angle feedback algorithm
        if( this->parameters_mc->metropolis_step_cone && this->parameters_mc->metropolis_cone_adaptive )
        {
            this->acceptance_ratio_current = 1 - (scalar)this->n_rejected / (scalar)this->nos_nonvacant;

            if( (this->acceptance_ratio_current < this->parameters_mc->acceptance_ratio_target) && (this->cone_angle > diff) )
                this->cone_angle -= diff;

            if( (this->acceptance_ratio_current > this->parameters_mc->acceptance_ratio_target) && (this->cone_angle < Constants::Pi-diff) )
                this->cone_angle += diff;

            this->parameters_mc->metropolis_cone_angle = this->cone_angle * 180.0 / Constants::Pi;
        }
        this->n_rejected = 0;

        // One Metropolis step for each spin
        Vector3 e_z{0, 0, 1};
        scalar costheta, sintheta, phi;
        Matrix3 local_basis;
        scalar cos_cone_angle = std::cos(cone_angle);

        // Loop over NOS samples (on average every spin should be hit once per Metropolis step)
        for (int idx=0; idx < this->nos; ++idx)
        {
            int ispin;
            if( this->parameters_mc->metropolis_random_sample )
                // Better statistics, but additional calculation of random number
                ispin = distribution_idx(this->parameters_mc->prng);
            else
                // Faster, but worse statistics
                ispin = idx;

            if( Vectormath::check_atom_type(this->systems[0]->geometry->atom_types[ispin]) )
            {
                // Sample a cone
                if( this->parameters_mc->metropolis_step_cone )
                {
                    // Calculate local basis for the spin
                    if( std::abs(spins_old[ispin].z()) < 1-1e-10 )
                    {
                        local_basis.col(2) = spins_old[ispin];
                        local_basis.col(0) = (local_basis.col(2).cross(e_z)).normalized();
                        local_basis.col(1) = local_basis.col(2).cross(local_basis.col(0));
                    }
                    else
                    {
                        local_basis = Matrix3::Identity();
                    }

                    // Rotation angle between 0 and cone_angle degrees
                    costheta = 1 - (1 - cos_cone_angle) * distribution(this->parameters_mc->prng);

                    sintheta = std::sqrt(1 - costheta*costheta);

                    // Random distribution of phi between 0 and 360 degrees
                    phi = 2*Constants::Pi * distribution(this->parameters_mc->prng);

                    // New spin orientation in local basis
                    Vector3 local_spin_new{ sintheta * std::cos(phi),
                                            sintheta * std::sin(phi),
                                            costheta };

                    // New spin orientation in regular basis
                    spins_new[ispin] = local_basis * local_spin_new;
                }
                // Sample the entire unit sphere
                else
                {
                    // Rotation angle between 0 and 180 degrees
                    costheta = distribution(this->parameters_mc->prng);

                    sintheta = std::sqrt(1 - costheta*costheta);

                    // Random distribution of phi between 0 and 360 degrees
                    phi = 2*Constants::Pi * distribution(this->parameters_mc->prng);

                    // New spin orientation in local basis
                    spins_new[ispin] = Vector3{ sintheta * std::cos(phi),
                                                sintheta * std::sin(phi),
                                                costheta };
                }

                // Energy difference of configurations with and without displacement
                scalar Eold  = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_old);
                scalar Enew  = this->systems[0]->hamiltonian->Energy_Single_Spin(ispin, spins_new);
                scalar Ediff = Enew-Eold;

                // Metropolis criterion: reject the step if energy rose
                if( Ediff > 1e-14 )
                {
                    if( this->parameters_mc->temperature < 1e-12 )
                    {
                        // Restore the spin
                        spins_new[ispin] = spins_old[ispin];
                        // Counter for the number of rejections
                        ++this->n_rejected;
                    }
                    else
                    {
                        // Exponential factor
                        scalar exp_ediff    = std::exp( -Ediff/kB_T );
                        // Metropolis random number
                        scalar x_metropolis = distribution(this->parameters_mc->prng);

                        // Only reject if random number is larger than exponential
                        if( exp_ediff < x_metropolis )
                        {
                            // Restore the spin
                            spins_new[ispin] = spins_old[ispin];
                            // Counter for the number of rejections
                            ++this->n_rejected;
                        }
                    }
                }
            }
        }
    }

    void Method_MC::Block_Decomposition()
    {
        /* 
        Decompose the system into blocks of spins
        Each Block may only interact with it's nearest neighbour.
        Therefore this decomposition only works for short-ranged interactions (all except DDI)
        It is further important that the number of blocks in each dimension is either 1 or an even number.
        If this is not fulfilled the block can interact with its nearest periodic neighbour, for open boundary conditions it does not matter.
        For example, a valid decomposition of a 9x4x1 system, with up to *next-nearest* neighbour interactions, could look like this:
            ┌───┬───┬───┬─────┐
            │1 1│2 2│1 1│2 2 2│
            │   │   │   │     │
            │1 1│2 2│1 1│2 2 2│
            ├───┼───┼───┼─────┤
            │3 3│4 4│3 3│4 4 4│
            │   │   │   │     │
            │3 3│4 4│3 3│4 4 4│
            └───┴───┴───┴─────┘
        The decomposition enables us to run the MC algorithm in four serial phases (1 to 4), while each phase is parallelized.
        All boxes belonging to one specific phase, do not interact directly between each other, therefore we can perform their MC trial moves in parallel.
        The number of phases is 2^dim and the number of usable threads is the number of boxes per phase.
        In this example the usable number of threads would be 2.
        If the size of the boxes does not perfectly divide n_cells, the rest is added or substracted at the edges.
        */

        // Initialize
        n_blocks = {0,0,0};
        block_size_min = {1,1,1}; // Require at least one as block size
        rest = {0,0,0};

        if (this->systems[0]->hamiltonian->Name() != "Heisenberg")
        {
            return;
        }

        Log.Send(Utility::Log_Level::Info, Utility::Log_Sender::MC, fmt::format("Performing block decomposition for parallel Metropolis algorithm"));
        const auto & geom = this->systems[0]->geometry;
        const auto & ham  = (const Hamiltonian_Heisenberg *) this->systems[0]->hamiltonian.get();
  
        for(auto & p : ham->dmi_pairs)
        {
            for(int i = 0; i < 3; i++)
                block_size_min[i] = std::max(block_size_min[i], std::abs(p.translations[i]));
        }
        for(auto & p : ham->exchange_pairs)
        {
            for(int i = 0; i < 3; i++)
                block_size_min[i] = std::max(block_size_min[i], std::abs(p.translations[i]));
        }
        for(auto & q : ham->quadruplets)
        {
            for(int i = 0; i < 3; i++)
            {
                block_size_min[i] = std::max(block_size_min[i], std::abs(q.d_j[i]));
                block_size_min[i] = std::max(block_size_min[i], std::abs(q.d_k[i]));
                block_size_min[i] = std::max(block_size_min[i], std::abs(q.d_l[i]));
            }
        }
        Log.Send(Utility::Log_Level::Info, Utility::Log_Sender::MC, fmt::format("    Block size from interactions = ({}, {}, {})", block_size_min[0], block_size_min[1], block_size_min[2]));
        
        // If the number of blocks is uneven for a certain direction, we increase the block size until it is even
        for(int i = 0; i < 3; i++)
        {
            n_blocks[i] = geom->n_cells[i] / block_size_min[i];
            while(n_blocks[i] % 2 != 0 )
            {
                n_blocks[i] = geom->n_cells[i] / ++block_size_min[i];
            }
        }

        for(int i = 0; i < 3; i++)
        {
            if(block_size_min[i] <= geom->n_cells[i])
            {
                rest[i] = geom->n_cells[i] % block_size_min[i];
            } else { // If the block size is larger than n_cells, we use n_blocks=1
                n_blocks[i] = 1;
                block_size_min[i] = geom->n_cells[i];
            }
        }

        Log.Send(Utility::Log_Level::Info, Utility::Log_Sender::MC, fmt::format("    Actual block size = ({}, {}, {})", block_size_min[0], block_size_min[1], block_size_min[2]));
        Log.Send(Utility::Log_Level::Info, Utility::Log_Sender::MC, fmt::format("    N_blocks          = ({}, {}, {})", n_blocks[0], n_blocks[1], n_blocks[2]));
        Log.Send(Utility::Log_Level::Info, Utility::Log_Sender::MC, fmt::format("    Rest              = ({}, {}, {})", rest[0], rest[1], rest[2]));
        max_supported_threads = std::max(1,n_blocks[0]/2) * std::max(1,n_blocks[1]/2) * std::max(1,n_blocks[2]/2);
        Log.Send(Utility::Log_Level::Info, Utility::Log_Sender::MC, fmt::format("    Supports up to max. {} threads.", max_supported_threads));
    }

    // Simple metropolis step
    #ifndef SPIRIT_USE_CUDA
    void Method_MC::Parallel_Metropolis(const vectorfield & spins_old, vectorfield & spins_new)
    {
        const auto & geom = this->systems[0]->geometry;

        scalar diff = 0.01;
        // Cone angle feedback algorithm
        if( this->parameters_mc->metropolis_step_cone && this->parameters_mc->metropolis_cone_adaptive )
        {
            this->acceptance_ratio_current = 1 - (scalar)this->n_rejected / (scalar)this->nos_nonvacant;

            if( (this->acceptance_ratio_current < this->parameters_mc->acceptance_ratio_target) && (this->cone_angle > diff) )
                this->cone_angle -= diff;

            if( (this->acceptance_ratio_current > this->parameters_mc->acceptance_ratio_target) && (this->cone_angle < Constants::Pi-diff) )
                this->cone_angle += diff;

            this->parameters_mc->metropolis_cone_angle = this->cone_angle * 180.0 / Constants::Pi;
        }

        this->n_rejected = 0;
        int n_rejected = 0;
        scalar cos_cone_angle = std::cos(this->cone_angle);

        auto distribution = std::uniform_real_distribution<scalar>(0, 1);

        // Number of threads
        int nt = 1;
        #ifdef SPIRIT_USE_OPENMP
            nt = omp_get_max_threads();
        #endif

        // We make nt copies of our number generator, so that each thread has its own
        if(nt != prng_vec.size())
        {
            prng_vec.resize(nt);
            for(int i=0; i<prng_vec.size(); i++)
            {
                prng_vec[i] = std::mt19937(this->parameters_mc->rng_seed * (i+1));
            }
        }

        // There are up to (2^dim) phases in the algorithm, which have to be executed serially
        for(int phase_c = 0; phase_c < 2; phase_c++)
        {
            for(int phase_b = 0; phase_b < 2; phase_b++)
            {
                for(int phase_a = 0; phase_a < 2; phase_a++)
                {
                    // In each phase we iterate over blocks of spins, which are **next-nearest** neighbour blocks
                    // Thus there is no dependence of the spin moves and we can parallelize over the blocks in a phase
                    #pragma omp parallel for collapse(3) private(distribution) reduction(+:n_rejected) 
                    for(int block_c = phase_c; block_c < n_blocks[2]; block_c += 2) // Note the += 2 increment!
                    {
                        for(int block_b = phase_b; block_b < n_blocks[1]; block_b += 2)
                        {
                            for(int block_a = phase_a; block_a < n_blocks[0]; block_a += 2)
                            {
                                int block_size_c = (block_c == n_blocks[2] - 1) ? block_size_min[2] + rest[2] : block_size_min[2]; // Account for the remainder of division (n_cells[i] / block_size_min[i]) by increasing the block size at the edges
                                int block_size_b = (block_b == n_blocks[1] - 1) ? block_size_min[1] + rest[1] : block_size_min[1];
                                int block_size_a = (block_a == n_blocks[0] - 1) ? block_size_min[0] + rest[0] : block_size_min[0];

                                // Iterate over the current block (this has to be done serially again)
                                for(int cc = 0; cc < block_size_c; cc++)
                                {
                                    for(int bb = 0; bb < block_size_b; bb++)
                                    {
                                        for(int aa = 0; aa < block_size_a; aa++)
                                        {
                                            for(int ibasis=0; ibasis < geom->n_cell_atoms; ibasis++)
                                            {
                                                int a = block_a * block_size_min[0] + aa; // We do not have to worry about the remainder of the division here, it is contained in the 'aa'/'bb'/'cc' offset
                                                int b = block_b * block_size_min[1] + bb;
                                                int c = block_c * block_size_min[2] + cc;

                                                // Compute the current spin idx
                                                int ispin = ibasis + geom->n_cell_atoms * (a + geom->n_cells[0] * (b + geom->n_cells[1] * c));

                                                // Perform the Metropolis trial step
                                                if( Vectormath::check_atom_type(this->systems[0]->geometry->atom_types[ispin]) )
                                                {
                                                    int tid = 0;
                                                    #ifdef SPIRIT_USE_OPENMP
                                                    tid = omp_get_thread_num();
                                                    #endif

                                                    scalar rng1 = distribution(prng_vec[tid]);
                                                    scalar rng2 = distribution(prng_vec[tid]);
                                                    scalar rng3 = distribution(prng_vec[tid]);
                                                    if( !Metropolis_Spin_Trial(ispin, spins_old, spins_new, rng1, rng2, rng3, cos_cone_angle) )
                                                    {
                                                        n_rejected++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        this->n_rejected = n_rejected;
        // std::cout << n_rejected << "\n";
        // Log.Send(Utility::Log_Level::Error, Utility::Log_Sender::MC, fmt::format( "n_rejected = {}",n_rejected ));
    }
    #endif

    // TODO:
    // Implement heat bath algorithm, see Y. Miyatake et al, J Phys C: Solid State Phys 19, 2539 (1986)
    // void Method_MC::HeatBath(const vectorfield & spins_old, vectorfield & spins_new)
    // {
    // }

    void Method_MC::Hook_Pre_Iteration()
    {
    }

    void Method_MC::Hook_Post_Iteration()
    {
    }

    #
    void Method_MC::Initialize()
    {
    }

    void Method_MC::Finalize()
    {
        this->systems[0]->iteration_allowed = false;
    }

    void Method_MC::Message_Start()
    {
        //---- Log messages
        std::vector<std::string> block(0);
        block.push_back(fmt::format("------------  Started  {} Calculation  ------------", this->Name()));
        block.push_back(fmt::format("    Going to iterate {} step(s)", this->n_log));
        block.push_back(fmt::format("                with {} iterations per step", this->n_iterations_log));
        if( this->parameters_mc->metropolis_step_cone )
        {
            if( this->parameters_mc->metropolis_cone_adaptive )
            {
                block.push_back(fmt::format("   Target acceptance {:>6.3f}", this->parameters_mc->acceptance_ratio_target));
                block.push_back(fmt::format("   Cone angle (deg): {:>6.3f} (adaptive)", this->cone_angle*180/Constants::Pi));
            }
            else
            {
                block.push_back(fmt::format("   Target acceptance {:>6.3f}", this->parameters_mc->acceptance_ratio_target));
                block.push_back(fmt::format("   Cone angle (deg): {:>6.3f} (non-adaptive)", this->cone_angle*180/Constants::Pi));
            }
        }
        block.push_back("-----------------------------------------------------");
        Log.SendBlock(Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain);
    }

    void Method_MC::Message_Step()
    {
        // Update time of current step
        auto t_current = system_clock::now();

        // Update the system's energy
        this->systems[0]->UpdateEnergy();

        // Send log message
        std::vector<std::string> block(0);
        block.push_back(fmt::format("----- {} Calculation: {}", this->Name(), Timing::DateTimePassed(t_current - this->t_start)));
        block.push_back(fmt::format("    Completed                 {} / {} step(s) (step size {})", this->step, this->n_log, this->n_iterations_log));
        block.push_back(fmt::format("    Iteration                 {} / {}", this->iteration, this->n_iterations));
        block.push_back(fmt::format("    Time since last step:     {}", Timing::DateTimePassed(t_current - this->t_last)));
        block.push_back(fmt::format("    Iterations / sec:         {}", this->n_iterations_log / Timing::SecondsPassed(t_current - this->t_last)));
        if( this->parameters_mc->metropolis_step_cone )
        {
            if( this->parameters_mc->metropolis_cone_adaptive )
            {
                block.push_back(fmt::format("    Current acceptance ratio: {:>6.3f} (target {})", this->acceptance_ratio_current, this->parameters_mc->acceptance_ratio_target));
                block.push_back(fmt::format("    Current cone angle (deg): {:>6.3f} (adaptive)", this->cone_angle*180/Constants::Pi));
            }
            else
            {
                block.push_back(fmt::format("    Current acceptance ratio: {:>6.3f}", this->acceptance_ratio_current));
                block.push_back(fmt::format("    Current cone angle (deg): {:>6.3f} (non-adaptive)", this->cone_angle*180/Constants::Pi));
            }
        }
        block.push_back(fmt::format("    Total energy:             {:20.10f}", this->systems[0]->E));
        Log.SendBlock(Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain);

        // Update time of last step
        this->t_last = t_current;
    }

    void Method_MC::Message_End()
    {
        //---- End timings
        auto t_end = system_clock::now();

        //---- Termination reason
        std::string reason = "";
        if( this->StopFile_Present() )
            reason = "A STOP file has been found";
        else if( this->Walltime_Expired(t_end - this->t_start) )
            reason = "The maximum walltime has been reached";

        // Update the system's energy
        this->systems[0]->UpdateEnergy();

        //---- Log messages
        std::vector<std::string> block;
        block.push_back(fmt::format("------------ Terminated {} Calculation ------------", this->Name()));
        if( reason.length() > 0 )
            block.push_back(fmt::format("----- Reason:   {}", reason));
        block.push_back(fmt::format("----- Duration:       {}", Timing::DateTimePassed(t_end - this->t_start)));
        block.push_back(fmt::format("    Completed         {} / {} step(s)", this->step, this->n_log));
        block.push_back(fmt::format("    Iteration         {} / {}", this->iteration, this->n_iterations));
        block.push_back(fmt::format("    Iterations / sec: {}", this->iteration / Timing::SecondsPassed(t_end - this->t_start)));
        if( this->parameters_mc->metropolis_step_cone )
        {
            if( this->parameters_mc->metropolis_cone_adaptive )
            {
                block.push_back(fmt::format("    Acceptance ratio: {:>6.3f} (target {})", this->acceptance_ratio_current, this->parameters_mc->acceptance_ratio_target));
                block.push_back(fmt::format("    Cone angle (deg): {:>6.3f} (adaptive)", this->cone_angle*180/Constants::Pi));
            }
            else
            {
                block.push_back(fmt::format("    Acceptance ratio: {:>6.3f}", this->acceptance_ratio_current));
                block.push_back(fmt::format("    Cone angle (deg): {:>6.3f} (non-adaptive)", this->cone_angle*180/Constants::Pi));
            }
        }
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