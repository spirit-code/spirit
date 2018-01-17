#include <Spirit_Defines.h>
#include <engine/Method_EMA.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Eigenmodes.hpp>
#include <data/Spin_System.hpp>
#include <io/IO.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <GenEigsSolver.h>  // Also includes <MatOp/DenseGenMatProd.h>
#include <GenEigsRealShiftSolver.h>

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

using namespace Utility;

namespace Engine
{
    /* helper function */
    void Check_n_modes(int& n_modes, const int nos, const int idx_img, const int idx_chain)
    {
        if (n_modes > 2*nos-2)
        {
            n_modes = 2*nos-2;
            Log(Log_Level::Warning, Log_Sender::EMA, fmt::format("Number of eigenmodes declared in "
            "EMA Parameters is too large. The number is set to {}", n_modes), idx_img, idx_chain);
        }
    }
    
    /* helper function */
    void Check_selected_mode(int& selected_mode, const int n_modes, const int idx_img, 
        const int idx_chain)
    {
        if (selected_mode > n_modes-1)
        {
            Log(Log_Level::Warning, Log_Sender::EMA, fmt::format("Eigenmode number {} is not "
            "available. The largest eigenmode ({}) is used instead", selected_mode, n_modes-1),
            idx_img, idx_chain);
            selected_mode = n_modes-1;
        }
    }
    
    void Calculate_Eigenmodes(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain)
    {
        int nos = system->nos;
        
        // vectorfield mode(nos, Vector3{1, 0, 0});
        vectorfield spins_initial = *system->spins;
        
        // get number of modes
        auto& n_modes = system->ema_parameters->n_modes; 
        
        // Check and set (if it is required) the total number of nodes
        Check_n_modes(n_modes, nos, idx_img, idx_chain);
        
        // Calculate the Eigenmodes
        vectorfield gradient(nos);
        MatrixX hessian(3*nos, 3*nos);
        
        // The gradient (unprojected)
        system->hamiltonian->Gradient(spins_initial, gradient);
        Vectormath::set_c_a(1, gradient, gradient, system->ema_parameters->pinning->mask_unpinned);
        
        // The Hessian (unprojected)
        system->hamiltonian->Hessian(spins_initial, hessian);
        
        // Get the eigenspectrum
        MatrixX hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
        MatrixX tangent_basis = MatrixX::Zero(3*nos, 2*nos);
        VectorX eigenvalues;
        MatrixX eigenvectors;
        bool successful = Eigenmodes::Hessian_Partial_Spectrum(spins_initial, gradient, hessian, 
            n_modes, tangent_basis, hessian_constrained, eigenvalues, eigenvectors);
        
        if (successful)
        {
            // get every mode and save it to system->modes
            for (int i=0; i<n_modes; i++)
            {
                // Extract the minimum mode (transform evec_lowest_2N back to 3N)
                VectorX evec_3N = tangent_basis * eigenvectors.col(i);
                
                // dynamically allocate the system->modes
                system->modes[i] = std::shared_ptr<vectorfield>(new vectorfield(nos, Vector3{1,0,0}));
                
                // Set the modes
                for (int j=0; j<nos; j++)
                    (*system->modes[i])[j] = {evec_3N[3*j], evec_3N[3*j+1], evec_3N[3*j+2]};
            }
        }
        else
        {
            //// TODO: What to do then?
        }
    }
    
    Method_EMA::Method_EMA(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
        Method(system->ema_parameters, idx_img, idx_chain)
    {
        // Currently we only support a single image being iterated at once:
        this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
        this->SenderName = Utility::Log_Sender::EMA;
        this->parameters_ema = system->ema_parameters;
        
        this->noi = this->systems.size();
        this->nos = this->systems[0]->nos;
        
        this->steps_per_period = 50;
        this->timestep = 1./this->steps_per_period;
        this->counter = 0;
        this->amplitude = 1;

        // attributes needed for applying a mode the spins
        this->angle = scalarfield(this->nos);
        this->angle_initial = scalarfield(this->nos);
        this->axis = vectorfield(this->nos);        
        this->spins_initial = *this->systems[0]->spins;
        
        // get number of modes and mode to visualize
        auto& n_modes = system->ema_parameters->n_modes;
        auto& selected_mode = system->ema_parameters->n_mode_follow;
        
        // Check and set (if it is required) the total number of nodes
        Check_n_modes(n_modes, nos, idx_img, idx_chain);
        
        // Check and set (if it is required) the mode to visualize
        Check_selected_mode(selected_mode, n_modes, idx_img, idx_chain);

        //// TODO: calculate the eigenmodes if they are not yet caluclated (check for NULL)
        Calculate_Eigenmodes(system, idx_img, idx_chain);

        // set the selected mode after checks and calculation (if needed)
        this->mode = *system->modes[selected_mode];

        // Find the axes of rotation for the mode to visualize
        for (int idx=0; idx<this->nos; idx++)
        {
            this->angle_initial[idx] = this->mode[idx].norm();
            this->axis[idx] = this->spins_initial[idx].cross(this->mode[idx]).normalized();
        }
        
        // for checking if the parameteres have been updated during iterations
        this->following_mode = this->parameters_ema->n_mode_follow;
    }
    
    void Method_EMA::Iteration()
    {
        // if the mode has change
        if ( this->following_mode != this->parameters_ema->n_mode_follow )
        {
            // reset local attribute for following mode
            this->following_mode = this->parameters_ema->n_mode_follow;
            // restore the initial spin configuration
            (*this->systems[0]->spins) = this->spins_initial;
            // set the new mode
            this->mode = *this->systems[0]->modes[following_mode];
            
            // Find the axes of rotation for the mode to visualize
            for (int idx=0; idx<this->nos; idx++)
            {
                this->angle_initial[idx] = this->mode[idx].norm();
                this->axis[idx] = this->spins_initial[idx].cross(this->mode[idx]).normalized();
            }
        }

        auto& image = *this->systems[0]->spins;

        // Calculate n for that iteration based on the initial n displacement vector
        scalar t_angle = this->amplitude * std::cos(2*M_PI*this->counter*this->timestep);
        this->angle = this->angle_initial;
        Vectormath::scale(this->angle, t_angle);

        // Rotate the spins
        Vectormath::rotate(this->spins_initial, this->axis, this->angle, image);
    }
    
    bool Method_EMA::Converged()
    {
        return false;
    }
    
    void Method_EMA::Save_Current(std::string starttime, int iteration, bool initial, bool final)
    {    
    }
    
    void Method_EMA::Hook_Pre_Iteration()
    {
    }
    
    void Method_EMA::Hook_Post_Iteration()
    {
        ++this->counter;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    void Method_EMA::Initialize()
    {
    }
    
    void Method_EMA::Finalize()
    {
        this->Lock();
        // The initial spin configuration must be restored
        (*this->systems[0]->spins) = this->spins_initial;
        this->Unlock();
    }
    
    void Method_EMA::Message_Start()
    {
        using namespace Utility;
        
        //---- Log messages
        Log.SendBlock(Log_Level::All, this->SenderName,
        {
            "------------  Started  " + this->Name() + " Calculation  ------------",
            "    Going to iterate " + fmt::format("{}", this->n_log) + " steps",
            "                with " + fmt::format("{}", this->n_iterations_log) + " iterations per step",
            "     Number of modes " + fmt::format("{}", this->parameters_ema->n_modes ),
            "-----------------------------------------------------"
        }, this->idx_image, this->idx_chain);
    }
    
    void Method_EMA::Message_Step()
    {
    }
    
    void Method_EMA::Message_End()
    {
    }
    
    // Method name as string
    std::string Method_EMA::Name() { return "EMA"; }
}