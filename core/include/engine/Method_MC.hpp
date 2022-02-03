#pragma once
#ifndef METHOD_MC_H
#define METHOD_MC_H

#include "Spirit_Defines.h"
#include <engine/Method.hpp>
// #include <engine/Method_Solver.hpp>
#include <data/Spin_System.hpp>
// #include <data/Parameters_Method_MC.hpp>

#include <vector>

// class crstate;
class curandStateWrapper;

namespace Engine
{
    /*
        The Monte Carlo method
    */
    class Method_MC : public Method
    {
    public:
        // Constructor
        Method_MC(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain);

        // Method name as string
        std::string Name() override;
        
    private:
        // Solver_Iteration represents one iteration of a certain Solver
        void Iteration() override;

        // Performs a trial move of a single spin
        bool Metropolis_Spin_Trial(int ispin, const vectorfield & spins_old, vectorfield & spins_new, const scalar & rng1, const scalar & rng2, const scalar & rng3, const scalar & cos_cone_angle);

        void Block_Decomposition();
        
        // Metropolis iteration with adaptive cone radius
        void Metropolis(const vectorfield & spins_old, vectorfield & spins_new);
        // Parallel MC
        void Parallel_Metropolis(const vectorfield & spins_old, vectorfield & spins_new);

        void Setup_Curand();
        curandStateWrapper * dev_random;

        // Save the current Step's Data: spins and energy
        void Save_Current(std::string starttime, int iteration, bool initial=false, bool final=false) override;
        // A hook into the Method before an Iteration of the Solver
        void Hook_Pre_Iteration() override;
        // A hook into the Method after an Iteration of the Solver
        void Hook_Post_Iteration() override;

        // Sets iteration_allowed to false for the corresponding method
        void Initialize() override;
        // Sets iteration_allowed to false for the corresponding method
        void Finalize() override;

        // Log message blocks
        void Message_Start() override;
        void Message_Step() override;
        void Message_End() override;


        std::shared_ptr<Data::Parameters_Method_MC> parameters_mc;

        // Cosine of current cone angle
        scalar cone_angle;
        int n_rejected;
        scalar acceptance_ratio_current;
        int nos_nonvacant;

        // Random vector array
        vectorfield xi;

        // Values for block decompositon
        field<int> block_size_min = {0,0,0};
        field<int> n_blocks = {0,0,0};
        field<int> rest = {0,0,0};
        int max_supported_threads;
        std::vector<std::mt19937> prng_vec = {};
    };
}

#endif