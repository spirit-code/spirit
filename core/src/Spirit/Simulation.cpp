#include <Spirit/Chain.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>

#include <data/State.hpp>
#include <engine/Method_EMA.hpp>
#include <engine/Method_GNEB.hpp>
#include <engine/Method_LLG.hpp>
#include <engine/Method_MC.hpp>
#include <engine/Method_MMF.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>

void free_run_info( Simulation_Run_Info info ) noexcept
{
    delete[] info.history_energy;
    delete[] info.history_iteration;
    delete[] info.history_max_torque;
};

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
        image->Lock();

        image->iteration_allowed  = true;
        image->singleshot_allowed = singleshot;

        if( n_iterations > 0 )
            image->mc_parameters->n_iterations = n_iterations;
        if( n_iterations_log > 0 )
            image->mc_parameters->n_iterations_log = n_iterations_log;

        auto method = std::shared_ptr<Engine::Method>( new Engine::Method_MC( image, idx_image, idx_chain ) );

        image->Unlock();

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
        image->Lock();

        image->iteration_allowed  = true;
        image->singleshot_allowed = singleshot;

        if( n_iterations > 0 )
            image->llg_parameters->n_iterations = n_iterations;
        if( n_iterations_log > 0 )
            image->llg_parameters->n_iterations_log = n_iterations_log;

        std::shared_ptr<Engine::Method> method;
        if( solver_type == int( Engine::Solver::SIB ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::SIB>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::Heun ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::Heun>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::Depondt ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::Depondt>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::RungeKutta4 ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::RungeKutta4>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::VP ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::VP>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::LBFGS_OSO ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::LBFGS_OSO>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::LBFGS_Atlas ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::LBFGS_Atlas>( image, idx_image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::VP_OSO ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::VP_OSO>( image, idx_image, idx_chain ) );
        else
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
                fmt::format( "Invalid solver_type {}", solver_type ) );

        image->Unlock();

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    int idx_image = -1;
    from_indices( state, idx_image, idx_chain, image, chain );

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
            chain->Lock();

            chain->iteration_allowed  = true;
            chain->singleshot_allowed = singleshot;

            if( n_iterations > 0 )
                chain->gneb_parameters->n_iterations = n_iterations;
            if( n_iterations_log > 0 )
                chain->gneb_parameters->n_iterations_log = n_iterations_log;

            std::shared_ptr<Engine::Method> method;
            if( solver_type == int( Engine::Solver::SIB ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::SIB>( chain, idx_chain ) );
            else if( solver_type == int( Engine::Solver::Heun ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::Heun>( chain, idx_chain ) );
            else if( solver_type == int( Engine::Solver::Depondt ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::Depondt>( chain, idx_chain ) );
            else if( solver_type == int( Engine::Solver::VP ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::VP>( chain, idx_chain ) );
            else if( solver_type == int( Engine::Solver::LBFGS_OSO ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::LBFGS_OSO>( chain, idx_chain ) );
            else if( solver_type == int( Engine::Solver::LBFGS_Atlas ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::LBFGS_Atlas>( chain, idx_chain ) );
            else if( solver_type == int( Engine::Solver::VP_OSO ) )
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::VP_OSO>( chain, idx_chain ) );
            else
                spirit_throw(
                    Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
                    fmt::format( "Invalid solver_type {}", solver_type ) );

            chain->Unlock();

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
        image->Lock();

        image->iteration_allowed  = true;
        image->singleshot_allowed = singleshot;

        if( n_iterations > 0 )
            image->mmf_parameters->n_iterations = n_iterations;
        if( n_iterations_log > 0 )
            image->mmf_parameters->n_iterations_log = n_iterations_log;

        std::shared_ptr<Engine::Method> method;
        if( solver_type == int( Engine::Solver::SIB ) )
            method = std::shared_ptr<Engine::Method>( new Engine::Method_MMF<Engine::Solver::SIB>( image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::Heun ) )
            method
                = std::shared_ptr<Engine::Method>( new Engine::Method_MMF<Engine::Solver::Heun>( image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::Depondt ) )
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_MMF<Engine::Solver::Depondt>( image, idx_chain ) );
        // else if (solver_type == int(Engine::Solver::NCG))
        //     method = std::shared_ptr<Engine::Method>(
        //         new Engine::Method_MMF<Engine::Solver::NCG>( image, idx_chain ) );
        else if( solver_type == int( Engine::Solver::VP ) )
            method = std::shared_ptr<Engine::Method>( new Engine::Method_MMF<Engine::Solver::VP>( image, idx_chain ) );
        else
            spirit_throw(
                Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning,
                fmt::format( "Invalid solver_type {}", solver_type ) );

        image->Unlock();

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
        image->Lock();

        image->iteration_allowed  = true;
        image->singleshot_allowed = singleshot;

        if( n_iterations > 0 )
            image->ema_parameters->n_iterations = n_iterations;
        if( n_iterations_log > 0 )
            image->ema_parameters->n_iterations_log = n_iterations_log;

        auto method = std::shared_ptr<Engine::Method>( new Engine::Method_EMA( image, idx_image, idx_chain ) );

        image->Unlock();

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
        method->Lock();
        for( int i = 0; i < N; i++ )
        {
            // Do one single Iteration
            method->Iteration();
            // Post-iteration hook
            method->Post_Iteration_Hook();

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
        // Unlock systems
        method->Unlock();
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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Determine wether to stop or start a simulation
    if( image->iteration_allowed )
    {
        // Currently iterating image, so we stop
        image->Lock();
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
        image->Unlock();
    }
    else if( chain->iteration_allowed )
    {
        // Currently iterating chain, so we stop
        chain->Lock();
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
        chain->Unlock();
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

scalar Simulation_Get_MaxTorqueComponent( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->getForceMaxAbsComponent();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->getForceMaxAbsComponent();
    }

    return 0;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Simulation_Get_Chain_MaxTorqueComponents( State * state, scalar * torques, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    int idx_image = -1;
    from_indices( state, idx_image, idx_chain, image, chain );
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

scalar Simulation_Get_MaxTorqueNorm( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    int idx_image = -1;
    from_indices( state, idx_image, idx_chain, image, chain );
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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->SolverName().c_str();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->SolverName().c_str();
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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( Simulation_Running_On_Image( state, idx_image, idx_chain ) )
    {
        if( state->method_image[idx_image] )
            return state->method_image[idx_image]->Name().c_str();
    }
    else if( Simulation_Running_On_Chain( state, idx_chain ) )
    {
        if( state->method_chain )
            return state->method_chain->Name().c_str();
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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    int idx_image = -1;
    from_indices( state, idx_image, idx_chain, image, chain );

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
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    int idx_image = -1;
    from_indices( state, idx_image, idx_chain, image, chain );

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