#include <Spirit/Parameters.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set LLG ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set LLG Output
void Parameters_Set_LLG_Output_Folder(State *state, const char * folder, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	image->Lock();
        image->llg_parameters->output_folder = folder;
    	image->Unlock();

    	Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             "Set LLG Output Folder = " + std::string(folder), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_Output_General( State *state, bool any, bool initial, bool final, 
                                        int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	image->Lock();
        image->llg_parameters->output_any = any;
        image->llg_parameters->output_initial = initial;
        image->llg_parameters->output_final = final;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_Output_Energy( State *state, bool energy_step, bool energy_archive, 
                                       bool energy_spin_resolved, bool energy_divide_by_nos, 
                                       int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
    	image->Lock();
        image->llg_parameters->output_energy_step = energy_step;
        image->llg_parameters->output_energy_archive = energy_archive;
        image->llg_parameters->output_energy_spin_resolved = energy_spin_resolved;
        image->llg_parameters->output_energy_divide_by_nspins = energy_divide_by_nos;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_Output_Configuration( State *state, bool configuration_step, 
                                              bool configuration_archive, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
    	image->Lock();
        image->llg_parameters->output_configuration_step = configuration_step;
        image->llg_parameters->output_configuration_archive = configuration_archive;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_N_Iterations( State *state, int n_iterations, int n_iterations_log, 
                                      int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
    	image->Lock();
        image->llg_parameters->n_iterations = n_iterations;
        image->llg_parameters->n_iterations_log = n_iterations_log;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}


// Set LLG Simulation Parameters
void Parameters_Set_LLG_Direct_Minimization( State *state, bool direct, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        from_indices(state, idx_image, idx_chain, image, chain);

    	image->Lock();
        auto p = image->llg_parameters;
        p->direct_minimization = direct;
    	image->Unlock();

        if (direct)
    	    Log( Utility::Log_Level::Info, Utility::Log_Sender::API, 
                 "Set LLG solver to direct minimization", idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "Set LLG solver to dynamics", 
                 idx_image, idx_chain );
         }
     catch( ... )
     {
         Utility::Handle_Exception( "", idx_image, idx_chain );
     }    
}

void Parameters_Set_LLG_Convergence(State *state, float convergence, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();
        auto p = image->llg_parameters;
        p->force_convergence = convergence;
        image->Unlock();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set LLG force convergence = {}", convergence), idx_image, idx_chain);
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_Time_Step(State *state, float dt, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	image->Lock();
        auto p = image->llg_parameters;
        p->dt = dt;
    	image->Unlock();

    	Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set LLG dt = {}", dt), idx_image, idx_chain);
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_Damping(State *state, float damping, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();
        auto p = image->llg_parameters;
        p->damping = damping;
        image->Unlock();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set LLG damping = {}", damping), idx_image, idx_chain);
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_Temperature(State *state, float T, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();

        image->llg_parameters->temperature = T;

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set LLG temperature to {}", T), idx_image, idx_chain);

        image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_LLG_STT( State *state, bool use_gradient, float magnitude, const float normal[3],
                             int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();

        // Gradient or monolayer
        image->llg_parameters->stt_use_gradient = use_gradient;
        // Magnitude
        image->llg_parameters->stt_magnitude = magnitude;
        // Normal
        image->llg_parameters->stt_polarisation_normal[0] = normal[0];
        image->llg_parameters->stt_polarisation_normal[1] = normal[1];
        image->llg_parameters->stt_polarisation_normal[2] = normal[2];
        if (image->llg_parameters->stt_polarisation_normal.norm() < 0.9)
        {
            image->llg_parameters->stt_polarisation_normal = { 0,0,1 };
            Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, 
                    "s_c_vec = {0,0,0} replaced by {0,0,1}" );
        }
        else 
            image->llg_parameters->stt_polarisation_normal.normalize();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set LLG spin current to {}, direction ({})", magnitude, image->llg_parameters->stt_polarisation_normal.transpose()), idx_image, idx_chain );
        if (use_gradient)
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "STT: using the gradient approximation", idx_image, idx_chain );
        else
            Log( Utility::Log_Level::Info, Utility::Log_Sender::API, "STT: using the pinned monolayer approximation", idx_image, idx_chain );

        image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}


/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set MC ------------------------------------------------------------ */
/*------------------------------------------------------------------------------------------------------ */

// Set MC Output
void Parameters_Set_MC_Output_Folder(State *state, const char * folder, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	image->Lock();
        image->mc_parameters->output_folder = folder;
    	image->Unlock();

    	Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             "Set MC Output Folder = " + std::string(folder), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_MC_Output_General( State *state, bool any, bool initial, bool final, 
                                       int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	image->Lock();
        image->mc_parameters->output_any = any;
        image->mc_parameters->output_initial = initial;
        image->mc_parameters->output_final = final;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_MC_Output_Energy( State *state, bool energy_step, bool energy_archive, 
                                      bool energy_spin_resolved, bool energy_divide_by_nos, 
                                      int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
    	image->Lock();
        image->mc_parameters->output_energy_step = energy_step;
        image->mc_parameters->output_energy_archive = energy_archive;
        image->mc_parameters->output_energy_spin_resolved = energy_spin_resolved;
        image->mc_parameters->output_energy_divide_by_nspins = energy_divide_by_nos;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_MC_Output_Configuration( State *state, bool configuration_step, 
                                             bool configuration_archive, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
    	image->Lock();
        image->mc_parameters->output_configuration_step = configuration_step;
        image->mc_parameters->output_configuration_archive = configuration_archive;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_MC_N_Iterations( State *state, int n_iterations, int n_iterations_log, 
                                     int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
            
    	image->Lock();
        image->mc_parameters->n_iterations = n_iterations;
        image->mc_parameters->n_iterations_log = n_iterations_log;
    	image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}


// Set MG Simulation Parameters
void Parameters_Set_MC_Temperature( State *state, float T, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();

        image->mc_parameters->temperature = T;

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set MC temperature to {}", T), idx_image, idx_chain);

        image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_MC_Acceptance_Ratio( State *state, float ratio, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();

        image->mc_parameters->acceptance_ratio_target = ratio;

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set MC acceptance ratio to {}", ratio), idx_image, idx_chain);

        image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set GNEB ---------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set GNEB Output
void Parameters_Set_GNEB_Output_Folder(State *state, const char * folder, int idx_chain) noexcept
{
	int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	chain->Lock();
        chain->gneb_parameters->output_folder = folder;
    	chain->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_Output_General( State *state, bool any, bool initial, bool final, int idx_chain ) noexcept
{
	int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
    
    	chain->Lock();
        chain->gneb_parameters->output_any = any;
        chain->gneb_parameters->output_initial = initial;
        chain->gneb_parameters->output_final = final;
    	chain->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_Output_Energies( State *state, bool energies_step, bool energies_interpolated, 
                                          bool energies_divide_by_nos, int idx_chain ) noexcept
{
	int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
            
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
    
    	chain->Lock();
        chain->gneb_parameters->output_energies_step = energies_step;
        chain->gneb_parameters->output_energies_interpolated = energies_interpolated;
        chain->gneb_parameters->output_energies_divide_by_nspins = energies_divide_by_nos;
    	chain->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_Output_Chain(State *state, bool chain_step, int idx_chain) noexcept
{
	int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	chain->Lock();
        chain->gneb_parameters->output_chain_step = chain_step;
    	chain->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_N_Iterations( State *state, int n_iterations, int n_iterations_log, 
                                       int idx_chain ) noexcept
{
	int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
    	chain->Lock();
        chain->gneb_parameters->n_iterations = n_iterations;
        chain->gneb_parameters->n_iterations_log = n_iterations_log;
    	chain->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

// Set GNEB Calculation Parameters
void Parameters_Set_GNEB_Convergence(State *state, float convergence, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        image->Lock();
        auto p = chain->gneb_parameters;
        p->force_convergence = convergence;
        image->Unlock();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set GNEB force convergence = {}", convergence), idx_image, idx_chain);
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_Spring_Constant( State *state, float spring_constant, 
                                          int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        chain->Lock();
        auto p = chain->gneb_parameters;
        p->spring_constant = spring_constant;
        chain->Unlock();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set GNEB spring constant = {}", spring_constant), idx_image, idx_chain);
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_Climbing_Falling(State *state, int image_type, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        chain->Lock();
        chain->image_type[idx_image] = static_cast<Data::GNEB_Image_Type>(image_type);
        chain->Unlock();

        Log(Utility::Log_Level::Info, Utility::Log_Sender::API,
            fmt::format("Set GNEB image type = {}", image_type), idx_image, idx_chain);
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Set_GNEB_Image_Type_Automatically(State *state, int idx_chain) noexcept
{
    int idx_image=-1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        for (int img = 1; img < chain->noi - 1; ++img)
        {
            scalar E0 = chain->images[img-1]->E;
            scalar E1 = chain->images[img]->E;
            scalar E2 = chain->images[img+1]->E;

            // Maximum
            if (E0 < E1 && E1 > E2) Parameters_Set_GNEB_Climbing_Falling(state, 1, img);
            // Minimum
            if (E0 > E1 && E1 < E2) Parameters_Set_GNEB_Climbing_Falling(state, 2, img);
        }
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}


/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get LLG ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get LLG Output Parameters
const char * Parameters_Get_LLG_Output_Folder(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        return image->llg_parameters->output_folder.c_str();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return nullptr;
    }
}

void Parameters_Get_LLG_Output_General( State *state, bool * any, bool * initial, 
                                        bool * final, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *any = image->llg_parameters->output_any;
        *initial = image->llg_parameters->output_initial;
        *final = image->llg_parameters->output_final;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_LLG_Output_Energy( State *state, bool * energy_step, bool * energy_archive, 
                                       bool * energy_spin_resolved, bool * energy_divide_by_nos, 
                                       int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *energy_step = image->llg_parameters->output_energy_step;
        *energy_archive = image->llg_parameters->output_energy_archive;
        *energy_spin_resolved = image->llg_parameters->output_energy_spin_resolved;
        *energy_divide_by_nos = image->llg_parameters->output_energy_divide_by_nspins;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_LLG_Output_Configuration( State *state, bool * configuration_step, 
                                              bool * configuration_archive, int idx_image, 
                                              int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *configuration_step = image->llg_parameters->output_configuration_step;
        *configuration_archive = image->llg_parameters->output_configuration_archive;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_LLG_N_Iterations( State *state, int * iterations, int * iterations_log, 
                                      int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = image->llg_parameters;
        *iterations = p->n_iterations;
        *iterations_log = p->n_iterations_log;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

// Get LLG Simulation Parameters
bool Parameters_Get_LLG_Direct_Minimization(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        from_indices(state, idx_image, idx_chain, image, chain);

        auto p = image->llg_parameters;
        return p->direct_minimization;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return false;
    }
}

float Parameters_Get_LLG_Convergence( State *state, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = image->llg_parameters;
        return (float)p->force_convergence;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

float Parameters_Get_LLG_Time_Step( State *state, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = image->llg_parameters;
        return (float)p->dt;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

float Parameters_Get_LLG_Damping(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = image->llg_parameters;
        return (float)p->damping;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

float Parameters_Get_LLG_Temperature(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        return (float)image->llg_parameters->temperature;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

void Parameters_Get_LLG_STT( State *state, bool * use_gradient, float * magnitude, float normal[3], 
                             int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
     
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        // Gradient or monolayer
        *use_gradient = image->llg_parameters->stt_use_gradient;

        // Magnitude
        *magnitude = (float)image->llg_parameters->stt_magnitude;
        // Normal
        normal[0] = (float)image->llg_parameters->stt_polarisation_normal[0];
        normal[1] = (float)image->llg_parameters->stt_polarisation_normal[1];
        normal[2] = (float)image->llg_parameters->stt_polarisation_normal[2];
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}


/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get MC ------------------------------------------------------------ */
/*------------------------------------------------------------------------------------------------------ */

// Get MC Output Parameters
const char * Parameters_Get_MC_Output_Folder(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
    
        return image->mc_parameters->output_folder.c_str();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return nullptr;
    }
}

void Parameters_Get_MC_Output_General( State *state, bool * any, bool * initial, bool * final, 
                                       int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
    
        *any = image->mc_parameters->output_any;
        *initial = image->mc_parameters->output_initial;
        *final = image->mc_parameters->output_final;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_MC_Output_Energy( State *state, bool * energy_step, bool * energy_archive, 
                                      bool * energy_spin_resolved, bool * energy_divide_by_nos, 
                                      int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *energy_step = image->mc_parameters->output_energy_step;
        *energy_archive = image->mc_parameters->output_energy_archive;
        *energy_spin_resolved = image->mc_parameters->output_energy_spin_resolved;
        *energy_divide_by_nos = image->mc_parameters->output_energy_divide_by_nspins;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_MC_Output_Configuration( State *state, bool * configuration_step, 
                                             bool * configuration_archive, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *configuration_step = image->mc_parameters->output_configuration_step;
        *configuration_archive = image->mc_parameters->output_configuration_archive;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_MC_N_Iterations( State *state, int * iterations, int * iterations_log, 
                                     int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = image->mc_parameters;
        *iterations = p->n_iterations;
        *iterations_log = p->n_iterations_log;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

// Get MC Simulation Parameters
float Parameters_Get_MC_Temperature(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        return (float)image->mc_parameters->temperature;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

float Parameters_Get_MC_Acceptance_Ratio(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        return (float)image->mc_parameters->acceptance_ratio_target;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get GNEB ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get GNEB Output Parameters
const char * Parameters_Get_GNEB_Output_Folder(State *state, int idx_chain) noexcept
{
    int idx_image = -1;

    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        return chain->gneb_parameters->output_folder.c_str();
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return nullptr;
    }
}

void Parameters_Get_GNEB_Output_General( State *state, bool * any, bool * initial, bool * final, 
                                         int idx_chain ) noexcept
{
    int idx_image = -1;

    try
    {    
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *any = chain->gneb_parameters->output_any;
        *initial = chain->gneb_parameters->output_initial;
        *final = chain->gneb_parameters->output_final;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_GNEB_Output_Energies( State *state, bool * energies_step, 
                                          bool * energies_interpolated, bool * energies_divide_by_nos, 
                                          int idx_chain ) noexcept
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        *energies_step = chain->gneb_parameters->output_energies_step;
        *energies_interpolated = chain->gneb_parameters->output_energies_interpolated;
        *energies_divide_by_nos = chain->gneb_parameters->output_energies_divide_by_nspins;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_GNEB_Output_Chain(State *state, bool * chain_step, int idx_chain) noexcept
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
    
        *chain_step = chain->gneb_parameters->output_chain_step;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

void Parameters_Get_GNEB_N_Iterations( State *state, int * iterations, int * iterations_log, 
                                       int idx_chain ) noexcept
{
	int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = chain->gneb_parameters;
        *iterations = p->n_iterations;
        *iterations_log = p->n_iterations_log;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
    }
}

// Get GNEB Calculation Parameters
float Parameters_Get_GNEB_Convergence(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = chain->gneb_parameters;
        return (float)p->force_convergence;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

float Parameters_Get_GNEB_Spring_Constant(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = chain->gneb_parameters;
        return (float)p->spring_constant;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

int Parameters_Get_GNEB_Climbing_Falling(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        return (int)chain->image_type[idx_image];
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}

int Parameters_Get_GNEB_N_Energy_Interpolations(State *state, int idx_chain) noexcept
{
    int idx_image = -1;
    
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;
        
        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );
        
        auto p = chain->gneb_parameters;
        return p->n_E_interpolations;
    }
    catch( ... )
    {
        spirit_handle_exception(idx_image, idx_chain);
        return 0;
    }
}
