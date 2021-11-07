#include <Spirit/Parameters_GNEB.h>

#include <data/State.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <algorithm>

#include <fmt/format.h>
#include <fmt/ostream.h>

using namespace Utility;

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Set GNEB ---------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Set GNEB Output
void Parameters_GNEB_Set_Output_Tag( State * state, const char * tag, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->gneb_parameters->output_file_tag = tag;
    chain->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set GNEB output tag = \"{}\"", tag ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_Output_Folder( State * state, const char * folder, int idx_chain ) noexcept
try
{
    int idx_image = -1;
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
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_Output_General( State * state, bool any, bool initial, bool final, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->gneb_parameters->output_any     = any;
    chain->gneb_parameters->output_initial = initial;
    chain->gneb_parameters->output_final   = final;
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_Output_Energies(
    State * state, bool energies_step, bool energies_interpolated, bool energies_divide_by_nos,
    bool energies_add_readability_lines, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->gneb_parameters->output_energies_step                  = energies_step;
    chain->gneb_parameters->output_energies_interpolated          = energies_interpolated;
    chain->gneb_parameters->output_energies_divide_by_nspins      = energies_divide_by_nos;
    chain->gneb_parameters->output_energies_add_readability_lines = energies_add_readability_lines;
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_Output_Chain( State * state, bool chain_step, int chain_filetype, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->gneb_parameters->output_chain_step  = chain_step;
    chain->gneb_parameters->output_vf_filetype = IO::VF_FileFormat( chain_filetype );
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_N_Iterations( State * state, int n_iterations, int n_iterations_log, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->gneb_parameters->n_iterations     = n_iterations;
    chain->gneb_parameters->n_iterations_log = n_iterations_log;
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

// Set GNEB Calculation Parameters
void Parameters_GNEB_Set_Convergence( State * state, float convergence, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    image->Lock();
    auto p               = chain->gneb_parameters;
    p->force_convergence = convergence;
    image->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format( "Set GNEB force convergence = {}", convergence ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_GNEB_Set_Spring_Constant( State * state, float spring_constant, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    auto p             = chain->gneb_parameters;
    p->spring_constant = spring_constant;
    chain->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format( "Set GNEB spring constant = {}", spring_constant ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_GNEB_Set_Spring_Force_Ratio( State * state, float ratio, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    auto p                = chain->gneb_parameters;
    ratio                 = std::max( std::min( ratio, 1.0f ), 0.0f );
    p->spring_force_ratio = ratio;
    chain->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format( "Set GNEB spring force ratio (E vs Rx) = {}", ratio ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_Path_Shortening_Constant(
    State * state, float path_shortening_constant, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    auto p                      = chain->gneb_parameters;
    p->path_shortening_constant = path_shortening_constant;
    chain->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API,
         fmt::format( "Set GNEB path shortening constant = {}", path_shortening_constant ), idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_Climbing_Falling( State * state, int image_type, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->image_type[idx_image] = static_cast<Data::GNEB_Image_Type>( image_type );
    chain->Unlock();

    Log( Utility::Log_Level::Parameter, Utility::Log_Sender::API, fmt::format( "Set GNEB image type = {}", image_type ),
         idx_image, idx_chain );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
}

void Parameters_GNEB_Set_Image_Type_Automatically( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    for( int img = 1; img < chain->noi - 1; ++img )
    {
        scalar E0 = chain->images[img - 1]->E;
        scalar E1 = chain->images[img]->E;
        scalar E2 = chain->images[img + 1]->E;

        // Maximum
        if( E0 < E1 && E1 > E2 )
            chain->image_type[img] = Data::GNEB_Image_Type::Climbing;
        // Minimum
        else if( E0 > E1 && E1 < E2 )
            chain->image_type[img] = Data::GNEB_Image_Type::Falling;
        else if( chain->image_type[img] != Data::GNEB_Image_Type::Stationary )
            chain->image_type[img] = Data::GNEB_Image_Type::Normal;
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Set_N_Energy_Interpolations( State * state, int n, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    chain->Lock();
    chain->gneb_parameters->n_E_interpolations = n;
    int size_interpolated                      = chain->noi + ( chain->noi - 1 ) * n;
    chain->Rx_interpolated                     = std::vector<scalar>( size_interpolated, 0 );
    chain->E_interpolated                      = std::vector<scalar>( size_interpolated, 0 );
    chain->E_array_interpolated = std::vector<std::vector<scalar>>( 7, std::vector<scalar>( size_interpolated, 0 ) );
    chain->Unlock();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

/*------------------------------------------------------------------------------------------------------ */
/*---------------------------------- Get GNEB ----------------------------------------------------------- */
/*------------------------------------------------------------------------------------------------------ */

// Get GNEB Output Parameters
const char * Parameters_GNEB_Get_Output_Tag( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return chain->gneb_parameters->output_file_tag.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return nullptr;
}

const char * Parameters_GNEB_Get_Output_Folder( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return chain->gneb_parameters->output_folder.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return nullptr;
}

void Parameters_GNEB_Get_Output_General(
    State * state, bool * any, bool * initial, bool * final, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    *any     = chain->gneb_parameters->output_any;
    *initial = chain->gneb_parameters->output_initial;
    *final   = chain->gneb_parameters->output_final;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Get_Output_Energies(
    State * state, bool * energies_step, bool * energies_interpolated, bool * energies_divide_by_nos,
    bool * energies_add_readability_lines, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    *energies_step                  = chain->gneb_parameters->output_energies_step;
    *energies_interpolated          = chain->gneb_parameters->output_energies_interpolated;
    *energies_divide_by_nos         = chain->gneb_parameters->output_energies_divide_by_nspins;
    *energies_add_readability_lines = chain->gneb_parameters->output_energies_add_readability_lines;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Get_Output_Chain( State * state, bool * chain_step, int * chain_filetype, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    *chain_step     = chain->gneb_parameters->output_chain_step;
    *chain_filetype = static_cast<int>( chain->gneb_parameters->output_vf_filetype );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

void Parameters_GNEB_Get_N_Iterations( State * state, int * iterations, int * iterations_log, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p          = chain->gneb_parameters;
    *iterations     = p->n_iterations;
    *iterations_log = p->n_iterations_log;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
}

// Get GNEB Calculation Parameters
float Parameters_GNEB_Get_Convergence( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = chain->gneb_parameters;
    return static_cast<float>( p->force_convergence );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

float Parameters_GNEB_Get_Spring_Constant( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = chain->gneb_parameters;
    return static_cast<float>( p->spring_constant );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

float Parameters_GNEB_Get_Spring_Force_Ratio( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = chain->gneb_parameters;
    return static_cast<float>( p->spring_force_ratio );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return 0;
}

float Parameters_GNEB_Get_Path_Shortening_Constant( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = chain->gneb_parameters;
    return static_cast<float>( p->path_shortening_constant );
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return 0;
}

int Parameters_GNEB_Get_Climbing_Falling( State * state, int idx_image, int idx_chain ) noexcept
try
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    return static_cast<int>( chain->image_type[idx_image] );
}
catch( ... )
{
    spirit_handle_exception_api( idx_image, idx_chain );
    return 0;
}

int Parameters_GNEB_Get_N_Energy_Interpolations( State * state, int idx_chain ) noexcept
try
{
    int idx_image = -1;
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;

    // Fetch correct indices and pointers
    from_indices( state, idx_image, idx_chain, image, chain );

    auto p = chain->gneb_parameters;
    return p->n_E_interpolations;
}
catch( ... )
{
    spirit_handle_exception_api( -1, idx_chain );
    return 0;
}