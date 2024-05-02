#include <engine/Manifoldmath.hpp>
#include <engine/Method.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Method.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>
#include <utility/Timing.hpp>

#include <algorithm>

using namespace Utility;

namespace Engine
{

template<typename system_t>
Method<system_t>::Method( std::shared_ptr<Data::Parameters_Method> parameters, int idx_img, int idx_chain )
        : iteration( 0 ),
          step( 0 ),
          idx_image( idx_img ),
          idx_chain( idx_chain ),
          n_iterations_amortize( parameters->n_iterations_amortize ),
          parameters( parameters )
{
    // Sender name for log messages
    this->SenderName = Log_Sender::All;

    this->history_iteration  = std::vector<int>();
    this->history_max_torque = std::vector<scalar>();
    this->history_energy     = std::vector<scalar>();

    // TODO: is this a good idea?
    this->n_iterations     = std::max( long( 1 ), this->parameters->n_iterations );
    this->n_iterations_log = std::min( this->parameters->n_iterations_log, this->n_iterations );
    if( this->n_iterations_log <= long( 0 ) )
        this->n_iterations_log = this->n_iterations;
    this->n_log = this->n_iterations / this->n_iterations_log;

    if( n_iterations_amortize > n_iterations_log )
    {
        n_iterations_amortize = n_iterations_log;
    }

    // Setup timings
    for( std::uint8_t i = 0; i < 7; ++i )
        this->t_iterations.push_back( std::chrono::system_clock::now() );
    this->ips       = 0;
    this->starttime = Timing::CurrentDateTime();

// Printing precision for scalars
#ifdef CORE_SCALAR_TYPE_FLOAT
    this->print_precision = 8;
#else
    this->print_precision = 12;
#endif
}

template<typename system_t>
void Method<system_t>::Iterate()
{
    //---- Start timings
    this->starttime = Timing::CurrentDateTime();
    this->t_start   = std::chrono::system_clock::now();
    auto t_current  = std::chrono::system_clock::now();
    this->t_last    = std::chrono::system_clock::now();

    //---- Log messages
    this->Message_Start();

    //---- Initial save
    this->Save_Current( this->starttime, this->iteration, true, false );

    //---- Iteration loop
    for( this->iteration = 0; this->ContinueIterating() && !this->Walltime_Expired( t_current - t_start );
         this->iteration += n_iterations_amortize )
    {
        t_current = std::chrono::system_clock::now();

        // Lock Systems
        this->Lock();

        // Pre-iteration hook
        this->Hook_Pre_Iteration();
        // Do n_iterations_amortize iterations
        for( int i = 0; i < n_iterations_amortize; i++ )
            this->Iteration();
        // Post-iteration hook
        this->Hook_Post_Iteration();

        // Recalculate FPS
        this->t_iterations.pop_front();
        this->t_iterations.push_back( std::chrono::system_clock::now() );

        // Log Output every n_iterations_log steps
        if( this->n_iterations_log > 0 && this->iteration > 0 && 0 == fmod( this->iteration, this->n_iterations_log ) )
        {
            ++this->step;
            this->Message_Step();
            this->Save_Current( this->starttime, this->iteration, false, false );
        }

        // Unlock systems
        this->Unlock();
    }

    //---- Finalize (set iterations_allowed to false etc.)
    this->Finalize();

    //---- Log messages
    this->step = this->iteration / this->n_iterations_log;
    this->Message_End();

    //---- Final save
    this->Save_Current( this->starttime, this->iteration, false, true );
}

template<typename system_t>
scalar Method<system_t>::getIterationsPerSecond()
{
    scalar l_ips = 0.0;
    for( std::size_t i = 0; i < t_iterations.size() - 1; ++i )
    {
        l_ips += Timing::SecondsPassed( t_iterations[i + 1] - t_iterations[i] );
    }
    this->ips = 1.0 / ( l_ips / ( t_iterations.size() - 1 ) ) * n_iterations_amortize;
    return this->ips;
}

template<typename system_t>
int Method<system_t>::getNIterations()
{
    return this->iteration;
}

template<typename system_t>
double Method<system_t>::get_simulated_time()
{
    // Not Implemented!
    spirit_throw(
        Utility::Exception_Classifier::Not_Implemented, Utility::Log_Level::Error,
        "Tried to use Method::get_simulated_time() of the Method base class!" );
}

template<typename system_t>
std::int64_t Method<system_t>::getWallTime()
{
    auto t_current                           = std::chrono::system_clock::now();
    std::chrono::duration<scalar> dt_seconds = t_current - this->t_start;
    auto dt_ms                               = std::chrono::duration_cast<std::chrono::milliseconds>( dt_seconds );
    return dt_ms.count();
}

template<typename system_t>
scalar Method<system_t>::getTorqueMaxNorm()
{
    return this->max_torque;
}

template<typename system_t>
std::vector<scalar> Method<system_t>::getTorqueMaxNorm_All()
{
    return { this->max_torque };
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////////////// Protected functions

template<typename system_t>
void Method<system_t>::Initialize() {}

template<typename system_t>
void Method<system_t>::Message_Start() {}
template<typename system_t>
void Method<system_t>::Message_Step() {}
template<typename system_t>
void Method<system_t>::Message_End() {}

template<typename system_t>
void Method<system_t>::Iteration() {}

template<typename system_t>
bool Method<system_t>::ContinueIterating()
{
    return this->iteration < this->n_iterations && this->Iterations_Allowed() && !this->StopFile_Present();
}

template<typename system_t>
bool Method<system_t>::Iterations_Allowed()
{
    return this->systems[0]->iteration_allowed;
}

template<typename system_t>
bool Method<system_t>::Walltime_Expired( std::chrono::duration<scalar> dt_seconds )
{
    if( this->parameters->max_walltime_sec <= 0 )
        return false;
    else
        return dt_seconds.count() > this->parameters->max_walltime_sec;
}

template<typename system_t>
bool Method<system_t>::StopFile_Present()
{
    std::ifstream f( "STOP" );
    return f.good();
}

template<typename system_t>
void Method<system_t>::Save_Current( std::string starttime, int iteration, bool initial, bool final )
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use Method::Save_Current() of the Method base class!" );
}

template<typename system_t>
void Method<system_t>::Hook_Pre_Iteration()
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use Method::Save_Current() of the Method base class!" );
}

template<typename system_t>
void Method<system_t>::Hook_Post_Iteration()
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use Method::Save_Current() of the Method base class!" );
}

template<typename system_t>
void Method<system_t>::Finalize()
{
    // Not Implemented!
    spirit_throw(
        Exception_Classifier::Not_Implemented, Log_Level::Error,
        "Tried to use Method::Save_Current() of the Method base class!" );
}

template<typename system_t>
void Method<system_t>::Lock()
{
    for( auto & system : this->systems )
        system->Lock();
}

template<typename system_t>
void Method<system_t>::Unlock()
{
    for( auto & system : this->systems )
        system->Unlock();
}

template<typename system_t>
std::string Method<system_t>::Name()
{
    // Not Implemented!
    Log( Log_Level::Error, Log_Sender::All, std::string( "Tried to use Method::Name() of the Method base class!" ) );
    return "--";
}

// Solver name as string
template<typename system_t>
std::string Method<system_t>::SolverName()
{
    // Not Implemented!
    Log( Log_Level::Error, Log_Sender::All,
         std::string( "Tried to use Method::SolverName() of the Method base class!" ), this->idx_image,
         this->idx_chain );
    return "--";
}

template<typename system_t>
std::string Method<system_t>::SolverFullName()
{
    // Not Implemented!
    Log( Log_Level::Error, Log_Sender::All,
         std::string( "Tried to use Method::SolverFullname() of the Method base class!" ), this->idx_image,
         this->idx_chain );
    return "--";
}

template class Method<Engine::Spin::system_t>;

} // namespace Engine
