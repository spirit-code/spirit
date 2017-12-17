#include <Spirit_Defines.h>
#include <engine/Method_EMA.hpp>
#include <engine/Vectormath.hpp>
#include <data/Spin_System.hpp>
#include <io/IO.hpp>
#include <utility/Logging.hpp>
#include <fmt/format.h>
#include <Eigen/Dense>

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
        
        this->spins_initial = *this->systems[0]->spins;
        this->axis = vectorfield(this->nos);
        this->mode = vectorfield(this->nos, Vector3{1,0,0});
        
        this->steps_per_period = 50;
        this->timestep = 1./this->steps_per_period;
        this->counter = 0;
        this->amplitude = 0.2;
        
        // Find the axes of rotation
        for (int idx=0; idx<nos; idx++)
            this->axis[idx] = spins_initial[idx].cross(this->mode[idx]).normalized();
    }
    
    void Method_EMA::Iteration()
    {
        int nos = this->systems[0]->spins->size();

        auto& image = *this->systems[0]->spins;

        // Calculate n for that iteration based on the initial n displacement vector
        scalar angle = this->amplitude * std::cos(2*M_PI*this->counter*this->timestep);

        // Rotate the spins
        for (int idx=0; idx<nos; idx++)
            Vectormath::rotate( this->spins_initial[idx], this->axis[idx], angle, image[idx] );

        // normalize all spins
        Vectormath::normalize_vectors( *this->systems[0]->spins );
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