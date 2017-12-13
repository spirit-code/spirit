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
        
        Vector3 dis_vec({0.1,0,0});
        this->n_init = dis_vec;
        this->n_iter = this->n_init;
        this->steps_per_period = 50;
        this->timestep = 1./this->steps_per_period;
        this->counter = 0;
        
    }
    
    void Method_EMA::Iteration()
    {
        int nos = this->systems[0]->spins->size();
        
        // auto& spins_initial = *this->systems[0]->spins;
        Vector3 spin;
        
        // calculate n for that iteration based on the initial n displacement vector
        this->n_iter = this->n_init * cos(2*M_PI*this->counter*this->timestep);
        
        for (int sp=0; sp<nos; sp++)
        {
            spin = (*this->systems[0]->spins)[sp];
            
            // find the axis of rotation
            this->axis = spin.cross(this->n_iter);
            this->axis /= this->axis.norm();
            
            // calculate the angle of rotation
            this->angle = atan2( this->n_iter.norm(), spin.norm() );
            
            // rotate S
            Vectormath::rotate( spin, this->axis, this->angle, spin );
            
            // set the new spin
            (*this->systems[0]->spins)[sp] = spin;
        }
        // normalize all spins
        Vectormath::normalize_vectors( *this->systems[0]->spins );
        
        // rotate the initial displacement vector
        
        //// XXX: Should we rotate also the initial vector
        // Vectormath::rotate( this->n_init, this->axis, this->angle, this->n_init );
        
        // // Spin displacement by addition
        // 
        // for (int sp=0; sp<nos; sp++)
        // {
        //     (*this->systems[0]->spins)[sp] += 
        //         this->n_init * cos(2*M_PI*this->counter*this->timestep);
        // }
        // 
        // Vectormath::normalize_vectors( *this->systems[0]->spins );
        
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