#include <Spirit/Parameters_MC.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set MC ------------------------------------------------------------ */
/*------------------------------------------------------------------------------------------------------ */

// Set MC Output
void Parameters_MC_Set_Output_Tag(State *state, const char * tag, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        image->Lock();
        image->mc_parameters->output_file_tag = tag;
        image->Unlock();

        Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
             fmt::format("Set MC output tag = \"{}\"", tag), idx_image, idx_chain );
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Set_Output_Folder(State *state, const char * folder, int idx_image, int idx_chain) noexcept
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Set_Output_General( State *state, bool any, bool initial, bool final, 
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Set_Output_Energy( State *state, bool energy_step, bool energy_archive, bool energy_spin_resolved,
                                      bool energy_add_readability_lines, bool energy_divide_by_nos, 
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
        image->mc_parameters->output_energy_add_readability_lines = energy_add_readability_lines;
        image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Set_Output_Configuration( State *state, bool configuration_step, bool configuration_archive,
                                             int configuration_filetype, int idx_image, int idx_chain ) noexcept
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
        image->mc_parameters->output_configuration_filetype = configuration_filetype;
        image->Unlock();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Set_N_Iterations( State *state, int n_iterations, int n_iterations_log, 
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}


// Set MG Simulation Parameters
void Parameters_MC_Set_Temperature( State *state, float T, int idx_image, int idx_chain ) noexcept
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Set_Acceptance_Ratio( State *state, float ratio, int idx_image, int idx_chain ) noexcept
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}


/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get MC ------------------------------------------------------------ */
/*------------------------------------------------------------------------------------------------------ */

// Get MC Output Parameters
const char * Parameters_MC_Get_Output_Tag(State *state, int idx_image, int idx_chain) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        return image->mc_parameters->output_file_tag.c_str();
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

const char * Parameters_MC_Get_Output_Folder(State *state, int idx_image, int idx_chain) noexcept
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return nullptr;
    }
}

void Parameters_MC_Get_Output_General( State *state, bool * any, bool * initial, bool * final, 
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Get_Output_Energy( State *state, bool * energy_step, bool * energy_archive,
                                      bool * energy_spin_resolved, bool * energy_divide_by_nos,
                                      bool * energy_add_readability_lines,
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
        *energy_add_readability_lines = image->mc_parameters->output_energy_add_readability_lines;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Get_Output_Configuration( State *state, bool * configuration_step, bool * configuration_archive,
                                             int * configuration_filetype, int idx_image, int idx_chain ) noexcept
{
    try
    {
        std::shared_ptr<Data::Spin_System> image;
        std::shared_ptr<Data::Spin_System_Chain> chain;

        // Fetch correct indices and pointers
        from_indices( state, idx_image, idx_chain, image, chain );

        *configuration_step = image->mc_parameters->output_configuration_step;
        *configuration_archive = image->mc_parameters->output_configuration_archive;
        *configuration_filetype = image->mc_parameters->output_configuration_filetype;
    }
    catch( ... )
    {
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

void Parameters_MC_Get_N_Iterations( State *state, int * iterations, int * iterations_log, 
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
        spirit_handle_exception_api(idx_image, idx_chain);
    }
}

// Get MC Simulation Parameters
float Parameters_MC_Get_Temperature(State *state, int idx_image, int idx_chain) noexcept
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}

float Parameters_MC_Get_Acceptance_Ratio(State *state, int idx_image, int idx_chain) noexcept
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
        spirit_handle_exception_api(idx_image, idx_chain);
        return 0;
    }
}