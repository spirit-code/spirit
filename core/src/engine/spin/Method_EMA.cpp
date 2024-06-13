#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Eigenmodes.hpp>
#include <engine/spin/Method_EMA.hpp>
#include <io/IO.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <Eigen/Dense>

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

using namespace Utility;
namespace C = Utility::Constants;

namespace Engine
{

namespace Spin
{

Method_EMA::Method_EMA( std::shared_ptr<system_t> system, int idx_img, int idx_chain )
        : Method( system->ema_parameters, idx_img, idx_chain )
{
    this->system         =  system;
    this->SenderName     = Log_Sender::EMA;
    this->parameters_ema = system->ema_parameters;

    this->noi = 1;
    this->nos = this->system->nos;

    // Attributes needed for applying a mode the spins
    this->angle         = scalarfield( this->nos );
    this->angle_initial = scalarfield( this->nos );
    this->axis          = vectorfield( this->nos );
    this->spins_initial = *this->system->state;

    Eigenmodes::Check_Eigenmode_Parameters( *system );

    // Calculate eigenmodes only in case that the selected mode to follow is not computed yet.
    if( !this->system->modes[this->parameters_ema->n_mode_follow].has_value() )
        Eigenmodes::Calculate_Eigenmodes( *system, idx_img, idx_chain );

    this->counter = 0;
    // Trigger update on first iteration
    this->following_mode = -1;
}

void Method_EMA::Iteration()
{
    // If the mode index has changed
    if( this->following_mode != this->parameters_ema->n_mode_follow )
    {
        this->counter = 0;

        // Re-check validity of parameters and system->modes
        Eigenmodes::Check_Eigenmode_Parameters( *this->system );

        // Reset members
        this->following_mode = this->parameters_ema->n_mode_follow;
        // Restore the initial spin configuration
        ( *this->system->state ) = this->spins_initial;

        // Set the new mode
        this->mode = *this->system->modes[following_mode];

        // Find the axes of rotation for the mode to visualize
        for( int idx = 0; idx < this->nos; idx++ )
        {
            this->angle_initial[idx] = this->mode[idx].norm();
            this->axis[idx]          = this->spins_initial[idx].cross( this->mode[idx] ).normalized();
        }
    }

    auto & image = *this->system->state;

    // Calculate n for that iteration based on the initial n displacement vector
    scalar t_angle;
    if( !this->parameters_ema->snapshot )
        t_angle
            = this->parameters_ema->amplitude * std::sin( 2 * C::Pi * this->counter * this->parameters_ema->frequency );
    else
        t_angle = this->parameters_ema->amplitude;

    this->angle = this->angle_initial;
    Vectormath::scale( this->angle, t_angle );

    // Rotate the spins
    Vectormath::rotate( this->spins_initial, this->axis, this->angle, image );

    ++this->counter;
    std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) );
}

void Method_EMA::Save_Current( std::string starttime, int iteration, bool initial, bool final ) {}

void Method_EMA::Hook_Pre_Iteration() {}

void Method_EMA::Hook_Post_Iteration() {}

void Method_EMA::Initialize() {}

void Method_EMA::Finalize()
{
    this->Lock();
    // The initial spin configuration must be restored
    ( *this->system->state ) = this->spins_initial;
    this->Unlock();
}

void Method_EMA::Message_Start()
{
    //---- Log messages
    Log( Log_Level::All, this->SenderName,
         {
             "------------  Started  " + this->Name() + " Visualization ------------",
             "    Mode frequency  " + fmt::format( "{}", this->parameters_ema->frequency ),
             "    Mode amplitude  " + fmt::format( "{}", this->parameters_ema->amplitude ),
             "    Number of modes " + fmt::format( "{}", this->parameters_ema->n_modes ),
             "-----------------------------------------------------",
         },
         this->idx_image, this->idx_chain );
}

void Method_EMA::Message_Step() {}

void Method_EMA::Message_End() {}

void Method_EMA::Lock(){
    this->system->Lock();
}

void Method_EMA::Unlock(){
    this->system->Unlock();
}

bool Method_EMA::Iterations_Allowed()
{
    return this->system->iteration_allowed;
}

// Method name as string
std::string Method_EMA::Name()
{
    return "EMA";
}

// Solver name as string
std::string Method_EMA::SolverName()
{
    return "None";
}

} // namespace Spin

} // namespace Engine
