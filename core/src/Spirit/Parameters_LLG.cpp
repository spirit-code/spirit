#include <Spirit/Parameters_LLG.h>

#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set LLG ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set LLG output
void Parameters_LLG_Set_Output_Tag( State * state, const char * tag, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    image->llg_parameters->output_file_tag = tag;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set LLG output tag = \"{}\"", tag ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Output_Folder( State * state, const char * folder, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    image->llg_parameters->output_folder = folder;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, "Set LLG Output Folder = " + std::string( folder ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Output_General(
    State * state, bool any, bool initial, bool final, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    image->llg_parameters->output_any     = any;
    image->llg_parameters->output_initial = initial;
    image->llg_parameters->output_final   = final;
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Output_Energy(
    State * state, bool energy_step, bool energy_archive, bool energy_spin_resolved, bool energy_divide_by_nos,
    bool energy_add_readability_lines, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    image->llg_parameters->output_energy_step                  = energy_step;
    image->llg_parameters->output_energy_archive               = energy_archive;
    image->llg_parameters->output_energy_spin_resolved         = energy_spin_resolved;
    image->llg_parameters->output_energy_divide_by_nspins      = energy_divide_by_nos;
    image->llg_parameters->output_energy_add_readability_lines = energy_add_readability_lines;
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Output_Configuration(
    State * state, bool configuration_step, bool configuration_archive, int configuration_filetype, int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    image->llg_parameters->output_configuration_step    = configuration_step;
    image->llg_parameters->output_configuration_archive = configuration_archive;
    image->llg_parameters->output_vf_filetype           = IO::VF_FileFormat( configuration_filetype );
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_N_Iterations(
    State * state, int n_iterations, int n_iterations_log, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    image->llg_parameters->n_iterations     = n_iterations;
    image->llg_parameters->n_iterations_log = n_iterations_log;
    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Set LLG Simulation Parameters
void Parameters_LLG_Set_Direct_Minimization( State * state, bool direct, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p                 = image->llg_parameters;
    p->direct_minimization = direct;
    image->Unlock();

    if( direct )
        Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, "Set LLG solver to direct minimization",
             idx_image, idx_chain );
    else
        Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, "Set LLG solver to dynamics", idx_image,
             idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Convergence( State * state, float convergence, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p               = image->llg_parameters;
    p->force_convergence = convergence;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format( "Set LLG force convergence = {}", convergence ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Time_Step( State * state, float dt, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p = image->llg_parameters;
    p->dt  = dt;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set LLG dt = {}", dt ), idx_image,
         idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Damping( State * state, float damping, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p     = image->llg_parameters;
    p->damping = damping;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set LLG damping = {}", damping ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Temperature( State * state, float T, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();

    image->llg_parameters->temperature = T;

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set LLG temperature to {}", T ),
         idx_image, idx_chain );

    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_Temperature_Gradient(
    State * state, float inclination, const float direction[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();

    Vector3 v_direction                                     = Vector3{ direction[0], direction[1], direction[2] };
    image->llg_parameters->temperature_gradient_inclination = inclination;
    image->llg_parameters->temperature_gradient_direction   = v_direction;

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format(
             "Set LLG temperature gradient to inclination={}, direction={}", inclination, v_direction.transpose() ),
         idx_image, idx_chain );

    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Set_STT(
    State * state, bool use_gradient, float magnitude, const float normal[3], int idx_image, int idx_chain ) noexcept
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
    if( image->llg_parameters->stt_polarisation_normal.norm() < 0.9 )
    {
        image->llg_parameters->stt_polarisation_normal = { 0, 0, 1 };
        Log( Utility::Log_Level::Warning, Utility::Log_Sender::API, "s_c_vec = {0,0,0} replaced by {0,0,1}" );
    }
    else
        image->llg_parameters->stt_polarisation_normal.normalize();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format(
             "Set LLG spin current to {}, direction ({})", magnitude,
             image->llg_parameters->stt_polarisation_normal.transpose() ),
         idx_image, idx_chain );
    if( use_gradient )
        Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, "STT: using the gradient approximation",
             idx_image, idx_chain );
    else
        Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, "STT: using the pinned monolayer approximation",
             idx_image, idx_chain );

    image->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get LLG ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get LLG Output Parameters
const char * Parameters_LLG_Get_Output_Tag( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return image->llg_parameters->output_file_tag.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

const char * Parameters_LLG_Get_Output_Folder( State * state, int idx_image, int idx_chain ) noexcept
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
    spirit_handle_exception_api( idx_image, idx_chain );
    return nullptr;
}

void Parameters_LLG_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    *any     = image->llg_parameters->output_any;
    *initial = image->llg_parameters->output_initial;
    *final   = image->llg_parameters->output_final;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Get_Output_Energy(
    State * state, bool * energy_step, bool * energy_archive, bool * energy_spin_resolved, bool * energy_divide_by_nos,
    bool * energy_add_readability_lines, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    *energy_step                  = image->llg_parameters->output_energy_step;
    *energy_archive               = image->llg_parameters->output_energy_archive;
    *energy_spin_resolved         = image->llg_parameters->output_energy_spin_resolved;
    *energy_divide_by_nos         = image->llg_parameters->output_energy_divide_by_nspins;
    *energy_add_readability_lines = image->llg_parameters->output_energy_add_readability_lines;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Get_Output_Configuration(
    State * state, bool * configuration_step, bool * configuration_archive, int * configuration_filetype, int idx_image,
    int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    *configuration_step     = image->llg_parameters->output_configuration_step;
    *configuration_archive  = image->llg_parameters->output_configuration_archive;
    *configuration_filetype = static_cast<int>( image->llg_parameters->output_vf_filetype );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Get_N_Iterations(
    State * state, int * iterations, int * iterations_log, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p          = image->llg_parameters;
    *iterations     = p->n_iterations;
    *iterations_log = p->n_iterations_log;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

// Get LLG Simulation Parameters
bool Parameters_LLG_Get_Direct_Minimization( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->llg_parameters;
    return p->direct_minimization;
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return false;
}

float Parameters_LLG_Get_Convergence( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->llg_parameters;
    return static_cast<float>( p->force_convergence );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

float Parameters_LLG_Get_Time_Step( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->llg_parameters;
    return static_cast<float>( p->dt );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

float Parameters_LLG_Get_Damping( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = image->llg_parameters;
    return static_cast<float>( p->damping );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

float Parameters_LLG_Get_Temperature( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return static_cast<float>( image->llg_parameters->temperature );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

void Parameters_LLG_Get_Temperature_Gradient(
    State * state, float * inclination, float direction[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Inclination
    *inclination = static_cast<float>( image->llg_parameters->temperature_gradient_inclination );
    // Direction
    direction[0] = static_cast<float>( image->llg_parameters->temperature_gradient_direction[0] );
    direction[1] = static_cast<float>( image->llg_parameters->temperature_gradient_direction[1] );
    direction[2] = static_cast<float>( image->llg_parameters->temperature_gradient_direction[2] );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_LLG_Get_STT(
    State * state, bool * use_gradient, float * magnitude, float normal[3], int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    // Gradient or monolayer
    *use_gradient = image->llg_parameters->stt_use_gradient;

    // Magnitude
    *magnitude = static_cast<float>( image->llg_parameters->stt_magnitude );
    // Normal
    normal[0] = static_cast<float>( image->llg_parameters->stt_polarisation_normal[0] );
    normal[1] = static_cast<float>( image->llg_parameters->stt_polarisation_normal[1] );
    normal[2] = static_cast<float>( image->llg_parameters->stt_polarisation_normal[2] );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}