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
    Method_EMA::Method_EMA(std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain) :
        Method(system->ema_parameters, idx_img, idx_chain)
    {
        // Currently we only support a single image being iterated at once:
        this->systems = std::vector<std::shared_ptr<Data::Spin_System>>(1, system);
        this->SenderName = Utility::Log_Sender::EMA;
        
        this->noi = this->systems.size();
        this->nos = this->systems[0]->nos;
        
        this->parameters_ema = system->ema_parameters;

        this->steps_per_period = 50;
        this->timestep = 1./this->steps_per_period;
        this->counter = 0;
        this->amplitude = 1;

        this->angle = scalarfield(this->nos);
        this->angle_initial = scalarfield(this->nos);

        this->spins_initial = *this->systems[0]->spins;
        this->mode = vectorfield(this->nos, Vector3{1, 0, 0});
        this->axis = vectorfield(this->nos);

        // Calculate the Eigenmodes
        int n_modes = 10;
        int selected_mode = 0;

        vectorfield gradient(this->nos);
        MatrixX hessian(3*this->nos, 3*this->nos);

        // The gradient (unprojected)
        system->hamiltonian->Gradient(spins_initial, gradient);
        Vectormath::set_c_a(1, gradient, gradient, this->parameters->pinning->mask_unpinned);

        // The Hessian (unprojected)
        system->hamiltonian->Hessian(spins_initial, hessian);

        // Get the eigenspectrum
        MatrixX hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
        MatrixX tangent_basis = MatrixX::Zero(3*nos, 2*nos);
        VectorX eigenvalues;
        MatrixX eigenvectors;
        bool successful = Eigenmodes::Hessian_Partial_Spectrum(spins_initial, gradient, hessian, n_modes, tangent_basis, hessian_constrained, eigenvalues, eigenvectors);

        if (successful)
        {
            // Extract the minimum mode (transform evec_lowest_2N back to 3N)
            VectorX evec_3N = tangent_basis * eigenvectors.col(selected_mode);

            // Set the mode
            for (int n=0; n<this->nos; ++n)
            {
                this->mode[n] = {evec_3N[3*n], evec_3N[3*n+1], evec_3N[3*n+2]};
                this->angle_initial[n] = this->mode[n].norm();
            }

            // Find the axes of rotation
            for (int idx=0; idx<nos; idx++)
                this->axis[idx] = spins_initial[idx].cross(this->mode[idx]).normalized();
        }
        else
        {
            // What to do then?
        }
    }
    
    void Method_EMA::Iteration()
    {
        int nos = this->systems[0]->spins->size();

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
        //// TODO: Needs proper implementation
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