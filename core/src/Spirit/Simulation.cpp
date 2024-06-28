#include <Spirit/Chain.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>

#include <data/State.hpp>
#include <engine/spin/Method_EMA.hpp>
#include <engine/spin/Method_GNEB.hpp>
#include <engine/spin/Method_LLG.hpp>
#include <engine/spin/Method_MC.hpp>
#include <engine/spin/Method_MMF.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>

void free_run_info( Simulation_Run_Info info ) noexcept
{
    delete[] info.history_energy;
    delete[] info.history_iteration;
    delete[] info.history_max_torque;
};

namespace
{

using Engine::Spin::Solver;

// alias for efficiently smuggling a constexpr value into a C++17 lambda
template<Solver solver>
using solver_tag = std::integral_constant<Solver, solver>;

} // namespace

// Helper function to start a simulation once a Method has been created
void run_method( Engine::Method & method, bool singleshot, Simulation_Run_Info * info = nullptr )
{
    if( singleshot )
    {
        //---- Start timings
        method.starttime = Utility::Timing::CurrentDateTime();
        method.t_start   = std::chrono::system_clock::now();
        method.t_last    = std::chrono::system_clock::now();
        method.iteration = 0;

        //---- Log messages
        method.Message_Start();

        //---- Initial save
        method.Save_Current( method.starttime, method.iteration, true, false );
    }
    else
    {
        method.Iterate();
        if( info != nullptr )
        {
            info->max_torque       = method.getTorqueMaxNorm();
            info->total_iterations = method.getNIterations();
            info->total_walltime   = method.getWallTime();
            info->total_ips        = scalar( info->total_iterations ) / info->total_walltime * 1000.0;

            if( !method.history_iteration.empty() )
            {
                info->n_history_iteration = method.history_iteration.size();
                info->history_iteration   = new int[method.history_iteration.size()];
                std::copy( method.history_iteration.begin(), method.history_iteration.end(), info->history_iteration );
            }

            if( !method.history_max_torque.empty() )
            {
                info->n_history_max_torque = method.history_max_torque.size();
                info->history_max_torque   = new scalar[method.history_max_torque.size()];
                std::copy(
                    method.history_max_torque.begin(), method.history_max_torque.end(), info->history_max_torque );
            }

            if( !method.history_energy.empty() )
            {
                info->n_history_energy = method.history_energy.size();
                info->history_energy   = new scalar[method.history_energy.size()];
                std::copy( method.history_energy.begin(), method.history_energy.end(), info->history_energy );
            }
        }
    }
}

void Simulation_MC_Start(
    State * state, int n_iterations, int n_iterations_log, bool singleshot, Simulation_Run_Info * info, int idx_image,
    int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else
    {
        // We are not iterating, so we create the Method and call Iterate
        std::shared_ptr<Engine::Method> method = [&, &img = image]
        {
            std::scoped_lock _{ *img };
            img->iteration_allowed  = true;
            img->singleshot_allowed = singleshot;

            if( n_iterations > 0 )
                img->mc_parameters->n_iterations = n_iterations;
            if( n_iterations_log > 0 )
                img->mc_parameters->n_iterations_log = n_iterations_log;

            return std::make_shared<Engine::Spin::Method_MC>( img, idx_image, idx_chain );
        }();

        state->method_image[idx_image] = method;
        run_method( *method, singleshot, info );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Simulation_LLG_Start(
    State * state, int solver_type, int n_iterations, int n_iterations_log, bool singleshot, Simulation_Run_Info * info,
    int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else
    {
        // We are not iterating, so we create the Method and call Iterate
        std::shared_ptr<Engine::Method> method = [solver_type, &img = image, singleshot, n_iterations, n_iterations_log,
                                                  idx_image, idx_chain]() -> std::shared_ptr<Engine::Method>
        {
            using Engine::Spin::Solver;
            using Engine::Spin::Method_LLG;

            // delay setting variables until after we have determined the solver
            const auto dispatcher = [&]( auto && tag )
            {
                std::scoped_lock _{ *img };

                img->iteration_allowed  = true;
                img->singleshot_allowed = singleshot;

                if( n_iterations > 0 )
                    img->llg_parameters->n_iterations = n_iterations;
                if( n_iterations_log > 0 )
                    img->llg_parameters->n_iterations_log = n_iterations_log;

                return std::make_shared<Method_LLG<std::decay_t<decltype( tag )>::value>>( img, idx_image, idx_chain );
            };

            switch( static_cast<Solver>( solver_type ) )
            {
                case Solver::SIB: return dispatcher( solver_tag<Solver::SIB>{} );
                case Solver::Heun: return dispatcher( solver_tag<Solver::Heun>{} );
                case Solver::Depondt: return dispatcher( solver_tag<Solver::Depondt>{} );
                case Solver::RungeKutta4: return dispatcher( solver_tag<Solver::RungeKutta4>{} );
                case Solver::VP: return dispatcher( solver_tag<Solver::VP>{} );
                case Solver::LBFGS_OSO: return dispatcher( solver_tag<Solver::LBFGS_OSO>{} );
                case Solver::LBFGS_Atlas: return dispatcher( solver_tag<Solver::LBFGS_Atlas>{} );
                case Solver::VP_OSO: return dispatcher( solver_tag<Solver::VP_OSO>{} );
                default:
                    spirit_throw(
                        Utility::Exception_Classifier::Unknown_Solver, Utility::Log_Level::Warning,
                        fmt::format( "Invalid solver_type {}", solver_type ) );
            }
        }();

        state->method_image[idx_image] = method;
        run_method( *method, singleshot, info );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Simulation_GNEB_Start(
    State * state, int solver_type, int n_iterations, int n_iterations_log, bool singleshot, Simulation_Run_Info * info,
    int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    int idx_image       = -1;
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.", -1,
                idx_chain ) );
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.", -1,
                idx_chain ) );
    }
    else
    {
        // We are not iterating, so we create the Method and call Iterate
        if( Simulation_Running_Anywhere_On_Chain( state, idx_chain ) )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                 std::string( "There are still one or more simulations running on the specified chain!" )
                     + std::string( " Please stop them before starting a GNEB calculation." ) );
        }
        else if( Chain_Get_NOI( state, idx_chain ) < 2 )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API,
                 std::string( "There are less than 2 images in the specified chain!" )
                     + std::string( " Please insert more before starting a GNEB calculation." ) );
        }
        else
        {
            std::shared_ptr<Engine::Method> method = [solver_type, &chn = chain, singleshot, n_iterations,
                                                      n_iterations_log, idx_chain]() -> std::shared_ptr<Engine::Method>
            {
                using Engine::Spin::Method_GNEB;
                using Engine::Spin::Solver;

                // delay setting variables until after we have determined the solver
                const auto dispatcher = [&]( auto && tag )
                {
                    std::scoped_lock _{ *chn };
                    chn->iteration_allowed  = true;
                    chn->singleshot_allowed = singleshot;

                    if( n_iterations > 0 )
                        chn->gneb_parameters->n_iterations = n_iterations;
                    if( n_iterations_log > 0 )
                        chn->gneb_parameters->n_iterations_log = n_iterations_log;

                    return std::make_shared<Method_GNEB<std::decay_t<decltype( tag )>::value>>( chn, idx_chain );
                };

                switch( static_cast<Solver>( solver_type ) )
                {
                    case Solver::SIB: return dispatcher( solver_tag<Solver::SIB>{} );
                    case Solver::Heun: return dispatcher( solver_tag<Solver::Heun>{} );
                    case Solver::Depondt: return dispatcher( solver_tag<Solver::Depondt>{} );
                    case Solver::VP: return dispatcher( solver_tag<Solver::VP>{} );
                    case Solver::LBFGS_OSO: return dispatcher( solver_tag<Solver::LBFGS_OSO>{} );
                    case Solver::LBFGS_Atlas: return dispatcher( solver_tag<Solver::LBFGS_Atlas>{} );
                    case Solver::VP_OSO: return dispatcher( solver_tag<Solver::VP_OSO>{} );
                    default:
                        spirit_throw(
                            Utility::Exception_Classifier::Unknown_Solver, Utility::Log_Level::Warning,
                            fmt::format( "Invalid solver_type {}", solver_type ) );
                }
            }();

            state->method_chain = method;
            run_method( *method, singleshot, info );
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Simulation_MMF_Start(
    State * state, int solver_type, int n_iterations, int n_iterations_log, bool singleshot, Simulation_Run_Info * info,
    int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else
    {
        // We are not iterating, so we create the Method and call Iterate
        std::shared_ptr<Engine::Method> method = [solver_type, &img = image, singleshot, n_iterations, n_iterations_log,
                                                  idx_chain]() -> std::shared_ptr<Engine::Method>
        {
            using Engine::Spin::Solver;
            using Engine::Spin::Method_MMF;

            // delay setting variables until after we have determined the solver
            const auto dispatcher = [&]( auto && tag )
            {
                std::scoped_lock _{ *img };
                img->iteration_allowed  = true;
                img->singleshot_allowed = singleshot;

                if( n_iterations > 0 )
                    img->mmf_parameters->n_iterations = n_iterations;
                if( n_iterations_log > 0 )
                    img->mmf_parameters->n_iterations_log = n_iterations_log;

                return std::make_shared<Method_MMF<std::decay_t<decltype( tag )>::value>>( img, idx_chain );
            };

            switch( static_cast<Solver>( solver_type ) )
            {
                case Solver::SIB: return dispatcher( solver_tag<Solver::SIB>{} );
                case Solver::Heun: return dispatcher( solver_tag<Solver::Heun>{} );
                case Solver::Depondt: return dispatcher( solver_tag<Solver::Depondt>{} );
                // case Solver::NCG: return dispatcher( solver_tag<Solver::NCG>{} );
                case Solver::VP: return dispatcher( solver_tag<Solver::VP>{} );
                default:
                    spirit_throw(
                        Utility::Exception_Classifier::Unknown_Solver, Utility::Log_Level::Warning,
                        fmt::format( "Invalid solver_type {}", solver_type ) );
            }
        }();

        state->method_image[idx_image] = method;
        run_method( *method, singleshot, info );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Simulation_EMA_Start(
    State * state, int n_iterations, int n_iterations_log, bool singleshot, Simulation_Run_Info * info, int idx_image,
    int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
                idx_image, idx_chain ) );
    }
    else
    {
        // We are not iterating, so we create the Method and call Iterate
        std::shared_ptr<Engine::Method> method = [&, &img = image]
        {
            std::scoped_lock _{ *img };
            img->iteration_allowed  = true;
            img->singleshot_allowed = singleshot;

            if( n_iterations > 0 )
                img->ema_parameters->n_iterations = n_iterations;
            if( n_iterations_log > 0 )
                img->ema_parameters->n_iterations_log = n_iterations_log;

            return std::make_shared<Engine::Spin::Method_EMA>( img, idx_image, idx_chain );
        }();

        state->method_image[idx_image] = method;

        run_method( *method, singleshot, info );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Simulation_SingleShot( State * state, int idx_image, int idx_chain ) noexcept
{
    Simulation_N_Shot( state, 1, idx_image, idx_chain );
}

void Simulation_N_Shot( State * state, int N, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Get Method pointer
    std::shared_ptr<Engine::Method> method = nullptr;
    if( image->iteration_allowed && image->singleshot_allowed )
        method = state->method_image[idx_image];
    else if( chain->iteration_allowed && chain->singleshot_allowed )
        method = state->method_chain;
    else
    {
        // No simulation has been started
        spirit_throw(
            Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
            fmt::format(
                "Tried to use Simulation_SingleShot on image {} of chain {} but no SingleShot simulation has been "
                "started.",
                idx_image, idx_chain ) );
    }

    // One Iteration
    auto t_current = std::chrono::system_clock::now();
    if( method->ContinueIterating() && !method->Walltime_Expired( t_current - method->t_start ) )
    {
        // Lock Systems
        std::scoped_lock _{ *method };

        for( int i = 0; i < N; i++ )
        {
            // Pre-iteration hook
            method->Hook_Pre_Iteration();
            // Do one single Iteration
            method->Iteration();
            // Post-iteration hook
            method->Hook_Post_Iteration();

            // Recalculate FPS
            method->t_iterations.pop_front();
            method->t_iterations.push_back( std::chrono::system_clock::now() );

            // Log Output every n_iterations_log steps
            bool log = false;
            if( method->n_iterations_log > 0 )
                log = method->iteration > 0 && 0 == fmod( method->iteration, method->n_iterations_log );
            if( log )
            {
                ++method->step;
                method->Message_Step();
                method->Save_Current( method->starttime, method->iteration, false, false );
            }
            ++method->iteration;
        }
    }

    // Check the conditions again after the iteration was performed,
    // as this condition may not be checked automatically (e.g. SingleShot
    // is not called anymore).
    t_current = std::chrono::system_clock::now();
    if( !method->ContinueIterating() || method->Walltime_Expired( t_current - method->t_start ) )
    {
        //---- Log messages
        method->step = method->iteration / method->n_iterations_log;
        method->Message_End();

        //---- Final save
        method->Save_Current( method->starttime, method->iteration, false, true );
        //---- Finalize (set iterations_allowed to false etc.)
        method->Finalize();

        if( image->singleshot_allowed )
            image->singleshot_allowed = false;
        if( chain->singleshot_allowed )
            chain->singleshot_allowed = false;
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Simulation_Stop( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image, so we stop
        std::scoped_lock _{ *image };
        image->iteration_allowed = false;
        if( image->singleshot_allowed )
        {
            image->singleshot_allowed = false;
            auto method               = state->method_image[idx_image];
            //---- Log messages
            method->step = method->iteration / method->n_iterations_log;
            method->Message_End();
            //---- Final save
            method->Save_Current( method->starttime, method->iteration, false, true );
            //---- Finalize (set iterations_allowed to false etc.)
            method->Finalize();
        }
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain, so we stop
        std::scoped_lock _{ *chain };
        chain->iteration_allowed = false;
        if( chain->singleshot_allowed )
        {
            auto method = state->method_chain;
            //---- Log messages
            method->step = method->iteration / method->n_iterations_log;
            method->Message_End();
            //---- Final save
            method->Save_Current( method->starttime, method->iteration, false, true );
            //---- Finalize (set iterations_allowed to false etc.)
            method->Finalize();
        }
    }
    else
    {
        // We are not iterating
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Simulation_Stop_All( State * state ) noexcept
try
{
    // GNEB and current image
    Simulation_Stop( state, -1, -1 );

    // LLG, MC, EMA, MMF
    for( int img = 0; img < state->chain->noi; ++img )
        Simulation_Stop( state, img, -1 );
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

scalar Simulation_Get_MaxTorqueNorm( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->getTorqueMaxNorm();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->getTorqueMaxNorm();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Simulation_Get_Chain_MaxTorqueNorms( State * state, scalar * torques, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    int idx_image       = -1;
    auto [image, chain] = from_indices( state, idx_image, idx_chain );
    throw_if_nullptr( torques, "torques" );

    if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        std::vector<scalar> t( chain->noi, 0 );

        if( state->method_chain )
            t = state->method_chain->getTorqueMaxNorm_All();

        for( int i = 0; i < chain->noi; ++i )
        {
            torques[i] = t[i];
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

scalar Simulation_Get_IterationsPerSecond( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->getIterationsPerSecond();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->getIterationsPerSecond();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Simulation_Get_Iteration( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->getNIterations();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->getNIterations();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

// Get time passed by the simulation in picoseconds
//		If an LLG simulation is running this returns the cumulatively summed dt.
//		Otherwise it returns 0.
scalar Simulation_Get_Time( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
        {
            if( state->method_image[idx_image]->Name() == "LLG" )
            {
                return state->method_image[idx_image]->get_simulated_time();
            }
        }
        return 0;
    }
    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Simulation_Get_Wall_Time( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->getWallTime();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->getWallTime();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

const char * Simulation_Get_Solver_Name( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return strdup( state->method_image[idx_image]->SolverName().data() );
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return strdup( state->method_chain->SolverName().data() );
    }

    return "";
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

const char * Simulation_Get_Method_Name( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return strdup( state->method_image[idx_image]->Name().data() );
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return strdup( state->method_chain->Name().data() );
    }

    return "";
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

bool Simulation_Running_On_Image( State * state, int idx_image, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return image->iteration_allowed;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}

bool Simulation_Running_On_Chain( State * state, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    int idx_image       = -1;
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    return state->chain->iteration_allowed;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return false;
}

bool Simulation_Running_Anywhere_On_Chain( State * state, int idx_chain ) noexcept
try
{
    // Fetch correct indices and pointers
    int idx_image       = -1;
    auto [image, chain] = from_indices( state, idx_image, idx_chain );

    if( Simulation_Running_On_Chain( state, idx_chain ) )
        return true;

    for( int i = 0; i < chain->noi; ++i )
        if( Simulation_Running_On_Image( state, i, idx_chain ) )
            return true;

    return false;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return false;
}
