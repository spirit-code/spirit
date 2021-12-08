#include <Spirit/Parameters_MMF.h>

#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <utility/Exception.hpp>

#include <memory>

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set MMF ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set MMF Output
void Parameters_MMF_Set_Output_Tag( State * state, const char * tag, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    auto p             = image->mmf_parameters;
    p->output_file_tag = tag;
    chain->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set MMF output tag = \"{}\"", tag ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Set_Output_Folder( State * state, const char * folder, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p           = image->mmf_parameters;
    p->output_folder = folder;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, "Set MMF Output Folder = " + std::string( folder ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Set_Output_General(
    State * state, bool any, bool initial, bool final, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p            = image->mmf_parameters;
    p->output_any     = any;
    p->output_initial = initial;
    p->output_final   = final;
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Set_Output_Energy(
    State * state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos,
    bool energy_add_readability_lines, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p                                 = image->mmf_parameters;
    p->output_energy_step                  = energy_step;
    p->output_energy_archive               = energy_archive;
    p->output_energy_spin_resolved         = energy_spin_resolved;
    p->output_energy_divide_by_nspins      = energy_divide_by_nos;
    p->output_energy_add_readability_lines = energy_add_readability_lines;
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Set_Output_Configuration(
    State * state, bool configuration_step, bool configuration_archive, int configuration_filetype, int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p                          = image->mmf_parameters;
    p->output_configuration_step    = configuration_step;
    p->output_configuration_archive = configuration_archive;
    p->output_vf_filetype           = IO::VF_FileFormat( configuration_filetype );
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Set_N_Iterations(
    State * state, int n_iterations, int n_iterations_log, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p              = image->mmf_parameters;
    p->n_iterations     = n_iterations;
    p->n_iterations_log = n_iterations_log;
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Set MMF Calculation Parameters
void Parameters_MMF_Set_N_Modes( State * state, int n_modes, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    if( n_modes < 1 || n_modes > 2 * image->nos )
    {
        Log( Utility::Log_Level::Debug, Utility::Log_Sender::API,
             fmt::format( "Illegal value of number of modes (max value is {})", 2 * image->nos ), idx_image,
             idx_chain );
    }
    else
    {
        image->Lock();
        auto p     = image->mmf_parameters;
        p->n_modes = n_modes;
        image->modes.resize( n_modes );
        p->n_mode_follow = std::min( p->n_mode_follow, n_modes );
        image->Unlock();

        Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
             fmt::format( "Set MMF number of modes = {}", n_modes ), idx_image, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Set_N_Mode_Follow( State * state, int n_mode_follow, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->mmf_parameters;
    if( n_mode_follow < 0
        || n_mode_follow
               > p->n_modes - 1 ) // ||
                                  //  n_mode_follow >= image->modes.size() || image->modes[n_mode_follow] == NULL )
    {
        Log( Utility::Log_Level::Debug, Utility::Log_Sender::API, fmt::format( "Illegal value of mode to follow" ),
             idx_image, idx_chain );
    }
    else
    {
        image->Lock();
        p->n_mode_follow = n_mode_follow;
        image->Unlock();

        Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
             fmt::format( "Set MMF mode to follow = {}", n_mode_follow ), idx_image, idx_chain );
    }
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get MMF ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get MMF Output Parameters
const char * Parameters_MMF_Get_Output_Tag( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->mmf_parameters;
    return p->output_file_tag.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

const char * Parameters_MMF_Get_Output_Folder( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->mmf_parameters;
    return p->output_folder.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

void Parameters_MMF_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p   = image->mmf_parameters;
    *any     = p->output_any;
    *initial = p->output_initial;
    *final   = p->output_final;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Get_Output_Energy(
    State * state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos,
    bool * energy_add_readability_lines, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p                        = image->mmf_parameters;
    *energy_step                  = p->output_energy_step;
    *energy_archive               = p->output_energy_archive;
    *energy_spin_resolved         = p->output_energy_spin_resolved;
    *energy_divide_by_nos         = p->output_energy_divide_by_nspins;
    *energy_add_readability_lines = p->output_energy_add_readability_lines;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Get_Output_Configuration(
    State * state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype, int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p                  = image->mmf_parameters;
    *configuration_step     = p->output_configuration_step;
    *configuration_archive  = p->output_configuration_archive;
    *configuration_filetype = static_cast<int>( p->output_vf_filetype );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_MMF_Get_N_Iterations(
    State * state, int * iterations, int * iterations_log, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p          = image->mmf_parameters;
    *iterations     = p->n_iterations;
    *iterations_log = p->n_iterations_log;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Get MMF Calculation Parameters
int Parameters_MMF_Get_N_Modes( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->mmf_parameters;
    return p->n_modes;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Parameters_MMF_Get_N_Mode_Follow( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->mmf_parameters;
    return p->n_mode_follow;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}