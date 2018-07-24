#include <Spirit/State.h>
#include <Spirit/Simulation.h>
#include <Spirit/Chain.h>

#include <data/State.hpp>
#include <engine/Method_LLG.hpp>
#include <engine/Method_MC.hpp>
#include <engine/Method_GNEB.hpp>
#include <engine/Method_EMA.hpp>
#include <engine/Method_MMF.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>


bool Get_Method( State *state, const char * c_method_type, const char * c_solver_type, 
                 int n_iterations, int n_iterations_log, bool singleshot,
                 int idx_image, int idx_chain, 
                 std::shared_ptr<Engine::Method> & method ) noexcept
try
{
    // Translate to string
    std::string method_type(c_method_type);
    std::string solver_type(c_solver_type);
    Engine::Solver solver;

    // Check validity of specified Method
    if (method_type == "LLG" || method_type == "GNEB" || method_type == "MMF")
    {
        // Determine the Solver kind
        if (solver_type == "SIB")
            solver = Engine::Solver::SIB;
        else if (solver_type == "Heun")
            solver = Engine::Solver::Heun;
        else if (solver_type == "Depondt")
            solver = Engine::Solver::Depondt;
        else if (solver_type == "NCG")
            solver = Engine::Solver::NCG;
        else if (solver_type == "VP")
            solver = Engine::Solver::VP;
        else
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, "Invalid Solver selected: " + 
                    solver_type);
            return false;
        }
    }

    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // ------ Nothing is iterating, so we could start a simulation ------

    // Lock the chain in order to prevent unexpected things
    chain->Lock();

    // Determine the method and chain(s) or image(s) involved
    if (method_type == "LLG")
    {
        image->iteration_allowed = true;
        image->singleshot_allowed = singleshot;
        if (n_iterations > 0) image->llg_parameters->n_iterations = n_iterations;
        if (n_iterations_log > 0) image->llg_parameters->n_iterations_log = n_iterations_log;

        if (solver == Engine::Solver::SIB)
            method = std::shared_ptr<Engine::Method>( 
                new Engine::Method_LLG<Engine::Solver::SIB>( image, idx_image, idx_chain ) );
        else if (solver == Engine::Solver::Heun)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::Heun>( image, idx_image, idx_chain ) );
        else if (solver == Engine::Solver::Depondt)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::Depondt>( image, idx_image, idx_chain ) );
        else if (solver == Engine::Solver::NCG)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::NCG>( image, idx_image, idx_chain ) );
        else if (solver == Engine::Solver::VP)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_LLG<Engine::Solver::VP>( image, idx_image, idx_chain ) );
    }
    else if (method_type == "MC")
    {
        image->iteration_allowed = true;
        image->singleshot_allowed = singleshot;
        method = std::shared_ptr<Engine::Method>(
            new Engine::Method_MC( image, idx_image, idx_chain ) );
    }
    else if (method_type == "EMA")
    {
        image->iteration_allowed = true;
        image->singleshot_allowed = singleshot;
        if (n_iterations > 0) image->ema_parameters->n_iterations = n_iterations;
        if (n_iterations_log > 0) image->ema_parameters->n_iterations_log = n_iterations_log;
        
        method = std::shared_ptr<Engine::Method>(
            new Engine::Method_EMA(image,idx_image,idx_chain));
    }
    else if (method_type == "GNEB")
    {
        if (Simulation_Running_Anywhere_On_Chain(state, idx_chain))
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, 
                    std::string( "There are still one or more simulations running on the specified chain!" ) +
                    std::string( " Please stop them before starting a GNEB calculation." ) );
            chain->Unlock();
            return false;
        }
        else if (Chain_Get_NOI(state, idx_chain) < 3)
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, 
                    std::string( "There are less than 3 images in the specified chain!" ) +
                    std::string( " Please insert more before starting a GNEB calculation." ) );
            chain->Unlock();
            return false;
        }
        else
        {
            chain->iteration_allowed = true;
            chain->singleshot_allowed = singleshot;
            if (n_iterations > 0) 
                chain->gneb_parameters->n_iterations = n_iterations;
            if (n_iterations_log > 0) 
                chain->gneb_parameters->n_iterations_log = n_iterations_log;

            if (solver == Engine::Solver::SIB)
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::SIB>( chain, idx_chain ) );
            else if (solver == Engine::Solver::Heun)
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::Heun>( chain, idx_chain ) );
            else if (solver == Engine::Solver::Depondt)
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::Depondt>( chain, idx_chain ) );
            else if (solver == Engine::Solver::NCG)
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::NCG>( chain, idx_chain ) );
            else if (solver == Engine::Solver::VP)
                method = std::shared_ptr<Engine::Method>(
                    new Engine::Method_GNEB<Engine::Solver::VP>( chain, idx_chain ) );
        }
    }
    else if (method_type == "MMF")
    {
        image->iteration_allowed = true;
        image->singleshot_allowed = singleshot;
        if (n_iterations > 0) 
            image->mmf_parameters->n_iterations = n_iterations;
        if (n_iterations_log > 0) 
            image->mmf_parameters->n_iterations_log = n_iterations_log;

        if (solver == Engine::Solver::SIB)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_MMF<Engine::Solver::SIB>( image, idx_chain ) );
        else if (solver == Engine::Solver::Heun)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_MMF<Engine::Solver::Heun>( image, idx_chain ) );
        else if (solver == Engine::Solver::Depondt)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_MMF<Engine::Solver::Depondt>( image, idx_chain ) );
        else if (solver == Engine::Solver::NCG)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_MMF<Engine::Solver::NCG>( image, idx_chain ) );
        else if (solver == Engine::Solver::VP)
            method = std::shared_ptr<Engine::Method>(
                new Engine::Method_MMF<Engine::Solver::VP>( image, idx_chain ) );
    }
    else
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::API, 
                "Invalid Method selected: " + method_type );
        chain->Unlock();
        return false;
    }

    // Create Simulation Information
    auto info = std::shared_ptr<Engine::Method>(method);

    // Add to correct list
    if (method_type == "LLG")
        state->method_image[idx_image] = info;
    else if (method_type == "MC")
    {
        state->method_image[idx_image] = info;
    }
    else if (method_type == "EMA")
        state->method_image[idx_image] = info;
    else if (method_type == "GNEB")
        state->method_chain = info;
    else if (method_type == "MMF")
        state->method_image[idx_image] = info;
    
    // Unlock chain in order to be able to iterate
    chain->Unlock();

    return true;
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);
    return false;
}


void Simulation_Start(State *state, const char * c_method_type, const char * c_solver_type,
    int n_iterations, int n_iterations_log, int idx_image, int idx_chain) noexcept
try
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    
    // Determine wether to stop or start a simulation
    if (image->iteration_allowed)
    {
        // Currently iterating image
        spirit_throw(Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning, fmt::format(
            "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
            idx_image, idx_chain));
    }
    else if (chain->iteration_allowed)
    {
        // Currently iterating chain
        spirit_throw(Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning, fmt::format(
            "Tried to use Simulation_Start on image {} of chain {}, but there is already a simulation running.",
            idx_image, idx_chain));
    }
    else
    {
        // We are not iterating, so we create the Method and call Iterate
        std::shared_ptr<Engine::Method> method;
        if( Get_Method( state, c_method_type, c_solver_type,
                        n_iterations, n_iterations_log, false,
                        idx_image, idx_chain, method ) )
            method->Iterate();
    }
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);        
}



void Simulation_Start_SingleShot(State *state, const char * c_method_type, const char * c_solver_type, 
    int n_iterations, int n_iterations_log, int idx_image, int idx_chain) noexcept
try
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    
    // Determine wether to stop or start a simulation
    if (image->iteration_allowed)
    {
        // Currently iterating image
        spirit_throw(Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning, fmt::format(
            "Tried to use Simulation_Start_SingleShot on image {} of chain {}, but there is already a simulation running.",
            idx_image, idx_chain));
    }
    else if (chain->iteration_allowed)
    {
        // Currently iterating chain
        spirit_throw(Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning, fmt::format(
            "Tried to use Simulation_Start_SingleShot on image {} of chain {}, but there is already a simulation running.",
            idx_image, idx_chain));
    }
    else
    {
        std::shared_ptr<Engine::Method> method;

        if( Get_Method( state, c_method_type, c_solver_type,
            n_iterations, n_iterations_log, true, idx_image, idx_chain, method) )
        {
            //---- Start timings
            method->starttime = Utility::Timing::CurrentDateTime();
            method->t_start = system_clock::now();
            auto t_current = system_clock::now();
            method->t_last = system_clock::now();
            method->iteration = 0;

            //---- Log messages
            method->Message_Start();

            //---- Initial save
            method->Save_Current(method->starttime, method->iteration, true, false);
        }
    }
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);        
}

void Simulation_SingleShot(State *state, int idx_image, int idx_chain) noexcept
try
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Get Method pointer
    std::shared_ptr<Engine::Method> method = nullptr;
    if( image->iteration_allowed )
        method = state->method_image[idx_image];
    else if( chain->iteration_allowed )
        method = state->method_chain;
    else
    {
        // No simulation has been started
        spirit_throw(Utility::Exception_Classifier::Unknown_Exception, Utility::Log_Level::Warning, fmt::format(
            "Tried to use Simulation_SingleShot on image {} of chain {}, but no SingleShot simulation has been started.",
            idx_image, idx_chain));
    }

    // One Iteration
    auto t_current = system_clock::now();
    if( method->ContinueIterating() &&
        !method->Walltime_Expired(t_current - method->t_start) )
    {

        // Lock Systems
        method->Lock();

        // Pre-iteration hook
        method->Hook_Pre_Iteration();
        // Do one single Iteration
        method->Iteration();
        // Post-iteration hook
        method->Hook_Post_Iteration();

        // Recalculate FPS
        method->t_iterations.pop_front();
        method->t_iterations.push_back(system_clock::now());

        // Log Output every n_iterations_log steps
        bool log = false;
        if (method->n_iterations_log > 0)
            log = method->iteration > 0 && 0 == fmod(method->iteration, method->n_iterations_log);
        if ( log )
        {
            ++method->step;
            method->Message_Step();
            method->Save_Current(method->starttime, method->iteration, false, false);
        }

        ++method->iteration;

        // Unlock systems
        method->Unlock();
    }

    // Check the conditions agein after the iteration was performed,
    // as this condition may not be checked automatically (e.g. SingleShot
    // is not called anymore).
    t_current = system_clock::now();
    if( !method->ContinueIterating() ||
        method->Walltime_Expired(t_current - method->t_start) )
    {
        image->iteration_allowed = false;
    }
}
catch( ... )
{
    spirit_handle_exception_api(idx_image, idx_chain);        
}



void Simulation_Stop(State *state, int idx_image, int idx_chain) noexcept
try
{
    // Fetch correct indices and pointers for image and chain
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    
    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );
    
    // Determine wether to stop or start a simulation
    if (image->iteration_allowed)
    {
        // Currently iterating image, so we stop
        image->Lock();
        image->iteration_allowed  = false;
        if( image->singleshot_allowed )
        {
            image->singleshot_allowed = false;
            auto method = state->method_image[idx_image];
            //---- Log messages
            method->Message_End();
            //---- Final save
            method->Save_Current(method->starttime, method->iteration, false, true);
            //---- Finalize (set iterations_allowed to false etc.)
            method->Finalize();
        }
        image->Unlock();
    }
    else if (chain->iteration_allowed)
    {
        // Currently iterating chain, so we stop
        chain->Lock();
        chain->iteration_allowed = false;
        if( chain->singleshot_allowed )
        {
            auto method = state->method_chain;
            //---- Log messages
            method->Message_End();
            //---- Final save
            method->Save_Current(method->starttime, method->iteration, false, true);
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
    spirit_handle_exception_api(idx_image, idx_chain);        
}



void Simulation_Stop_All(State *state) noexcept
try
{
    // GNEB and current image
    Simulation_Stop(state, -1, -1);

    // LLG, MC, EMA, MMF
    for (int img = 0; img < state->chain->noi; ++img)
        Simulation_Stop(state, img, -1);
}
catch( ... )
{
    spirit_handle_exception_api(-1, -1);
}


float Simulation_Get_MaxTorqueComponent(State * state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float) state->method_image[idx_image]->getForceMaxAbsComponent();
        }
        else if (Simulation_Running_On_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float) state->method_chain->getForceMaxAbsComponent();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}


void Simulation_Get_Chain_MaxTorqueComponents(State * state, float * torques, int idx_chain) noexcept
{
    int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        if (Simulation_Running_On_Chain(state, idx_chain))
        {
            std::vector<scalar> t(chain->noi, 0);
            
            if (state->method_chain)
                t = state->method_chain->getForceMaxAbsComponent_All();

            for (int i=0; i<chain->noi; ++i)
            {
                torques[i] = t[i];
            }
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}


float Simulation_Get_IterationsPerSecond(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float)state->method_image[idx_image]->getIterationsPerSecond();
        }
        else if (Simulation_Running_On_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float)state->method_chain->getIterationsPerSecond();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}


int Simulation_Get_Iteration(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float)state->method_image[idx_image]->getNIterations();
        }
        else if (Simulation_Running_On_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float)state->method_chain->getNIterations();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

// Get time passed by the simulation in picoseconds
//		If an LLG simulation is running this returns the cumulatively summed dt.
//		Otherwise it returns 0.
float Simulation_Get_Time(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
            {
                if (state->method_image[idx_image]->Name() == "LLG")
                    return (float)state->method_image[idx_image]->getTime();
            }
            return 0;
        }
        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

int Simulation_Get_Wall_Time(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return (float)state->method_image[idx_image]->getWallTime();
        }
        else if (Simulation_Running_On_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return (float)state->method_chain->getWallTime();
        }

        return 0;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

const char * Simulation_Get_Solver_Name(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        
        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return state->method_image[idx_image]->SolverName().c_str();
        }
        else if (Simulation_Running_On_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return state->method_chain->SolverName().c_str();
        }
        
        return "";
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

const char * Simulation_Get_Method_Name(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_On_Image(state, idx_image, idx_chain))
        {
            if (state->method_image[idx_image])
                return state->method_image[idx_image]->Name().c_str();
        }
        else if (Simulation_Running_On_Chain(state, idx_chain))
        {
            if (state->method_chain)
                return state->method_chain->Name().c_str();
        }

        return "";
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}



bool Simulation_Running_On_Image(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (image->iteration_allowed) return true;
        else return false;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}

bool Simulation_Running_On_Chain(State *state, int idx_chain) noexcept
{
    int idx_image=-1;

    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
            from_indices( state, idx_image, idx_chain, image, chain );
        
        if (state->chain->iteration_allowed)
            return true;
        else 
            return false;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}

bool Simulation_Running_Anywhere_On_Chain(State *state, int idx_chain) noexcept
{
    int idx_image=-1;

    try
    {
        // Fetch correct indices and pointers for image and chain
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        if (Simulation_Running_On_Chain(state, idx_chain)) 
            return true;
        
        for (int i=0; i<chain->noi; ++i)
            if (Simulation_Running_On_Image(state, i, idx_chain)) 
                return true;
        
        return false;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return false;
    }
}