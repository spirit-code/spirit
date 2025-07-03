#include <Spirit/Spirit_Defines.h>
#include <Spirit/State.h>

#include <data/State.hpp>
#include <filesystem>
#include <io/Configconverter.hpp>
#include <io/IO.hpp>
#include <utility/Configuration_Chain.hpp>
#include <utility/Configurations.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <fmt/format.h>

using namespace Utility;

namespace
{

namespace detail
{

class Line : public std::string
{
    friend std::istream & operator>>( std::istream & is, Line & line )
    {
        return std::getline( is, line );
    }
};

} // namespace detail

std::string escape_lines( const std::string & comment, const char comment_char = '#' )
{
    std::istringstream iss( comment );
    const bool is_commented = std::all_of(
        std::istream_iterator<detail::Line>( iss ), std::istream_iterator<detail::Line>(),
        [comment_char]( const std::string & line )
        {
            if( line.empty() )
                return true;
            const auto first_non_wspace = std::find_if_not( line.begin(), line.end(), std::iswspace );
            return first_non_wspace == line.end() || *first_non_wspace == comment_char;
        } );

    if( !is_commented )
    {
        std::istringstream iss_( comment );
        std::ostringstream oss{};
        for( std::string line; std::getline( iss_, line ); )
            oss << comment_char << ( line.empty() ? "" : " " ) << line << '\n';
        return oss.str();
    }
    else
    {
        return comment;
    }
}

} // namespace

// Forward declaration of helper function
void Save_Initial_Final( State * state, bool initial ) noexcept;

State * State_Setup( const char * config_file, bool quiet ) noexcept
try
{
    //---------------------- Initial state data ------------------------------------------------
    auto * state = new State();

    // Initialize the state
    state->datetime_creation        = std::chrono::system_clock::now();
    state->datetime_creation_string = fmt::format( "{:%Y-%m-%d_%H-%M-%S}", state->datetime_creation );
    state->config_file              = config_file;
    state->quiet                    = quiet;

    // Check if config file exists
    if( !state->config_file.empty() )
    {
        try
        {
            IO::Filter_File_Handle myfile( state->config_file );
        }
        catch( ... )
        {
            Log( Log_Level::Error, Log_Sender::All,
                 fmt::format( "Could not find config file \"{}\"", state->config_file ) );
            state->config_file = "";
        }
    }
    //------------------------------------------------------------------------------------------

    //---------------------- Initial block of log messages -------------------------------------
    std::vector<std::string> block;
    block.emplace_back( "=====================================================" );
    block.emplace_back( "========== Spirit State: Initialising... ============" );

    // Log version info
    block.emplace_back( fmt::format( "==========     Version:  {}", version ) );
    // Log revision hash
    block.emplace_back( fmt::format( "==========     Revision: {}", version_revision ) );
    Log( Log_Level::All, Log_Sender::All, block );
    // Log compiler
    Log( Log_Level::Info, Log_Sender::All, fmt::format( "==========     Compiled with: {}", compiler_full ) );

    // Log whether running in "quiet" mode
    if( state->quiet )
        Log( Log_Level::All, Log_Sender::All, "Going to run in QUIET mode (only Error messages, no output files)" );

    // Log config file info
    if( !state->config_file.empty() )
        Log( Log_Level::All, Log_Sender::All, fmt::format( "Config file: \"{}\"", state->config_file ) );
    else
        Log( Log_Level::All, Log_Sender::All, "No config file. Will use default parameters." );
    //------------------------------------------------------------------------------------------
    const auto tbl      = IO::TOML_from_Config( state->config_file );
    const auto defaults = IO::Defaults_from_TOML( tbl );
    //---------------------- Initialize the log ------------------------------------------------
    try
    {
        IO::Log_from_TOML( tbl, defaults, state->quiet );
    }
    catch( ... )
    {
        spirit_handle_exception_api( -1, -1 );
    }
    //------------------------------------------------------------------------------------------

    //----------------------- Additional info log ----------------------------------------------
    block.clear();
    block.emplace_back( "=====================================================" );
    block.emplace_back( "========== Optimization Info" );
// Log OpenMP info
#ifdef SPIRIT_USE_OPENMP
    int nt = omp_get_max_threads();
    block.emplace_back( fmt::format( "Using OpenMP (max. {} threads)", nt ) );
#else
    block.emplace_back( "    Not using OpenMP" );
#endif
// Log STL parallelization info
#ifdef SPIRIT_USE_STDPAR
    block.emplace_back( "    Using parallel STL" );
#else
    block.emplace_back( "    Not using parallel STL" );
#endif
// Log CUDA info
#ifdef SPIRIT_USE_CUDA
    block.emplace_back( "    Using CUDA" );
#else
    block.emplace_back( "    Not using CUDA" );
#endif
// Log threading info
#ifdef SPIRIT_USE_THREADS
    block.emplace_back( "    Using std::thread" );
#else
    block.emplace_back( "    Not using std::thread" );
#endif
// Log defects info
#ifdef SPIRIT_ENABLE_DEFECTS
    block.emplace_back( "    Defects are enabled" );
#else
    block.emplace_back( "    Defects are not enabled" );
#endif
// Log pinning info
#ifdef SPIRIT_ENABLE_PINNING
    block.emplace_back( "    Pinning is enabled" );
#else
    block.emplace_back( "    Pinning is not enabled" );
#endif
// Log Precision info
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
    block.emplace_back( "    Using double as scalar type" );
#endif
#ifdef SPIRIT_SCALAR_TYPE_FLOAT
    block.emplace_back( "    Using float as scalar type" );
#endif
    Log( Log_Level::Info, Log_Sender::All, block );
    Log( Log_Level::All, Log_Sender::All, "=====================================================" );
    //------------------------------------------------------------------------------------------

    using Engine::Field;
    using Engine::get;
    //---------------------- Initialize spin_system --------------------------------------------
    state->active_image = IO::Spin_System_from_TOML( tbl, defaults );
    auto & image        = state->active_image;
    Configurations::Random_Sphere(
        image->state->spin, image->hamiltonian->get_geometry(), image->llg_parameters->prng );
    //------------------------------------------------------------------------------------------

    //----------------------- Initialize spin system chain -------------------------------------
    // Get parameters
    auto params_gneb
        = std::shared_ptr<Data::Parameters_Method_GNEB>( IO::Parameters_Method_GNEB_from_TOML( tbl, defaults ) );

    // Create the chain
    auto sv = std::vector<std::shared_ptr<State::system_t>>();
    sv.push_back( state->active_image );
    state->chain = std::make_shared<State::chain_t>( sv, params_gneb, false );
    //------------------------------------------------------------------------------------------

    //----------------------- Fill in the state ------------------------------------------------
    // Active image
    state->idx_active_image = 0;

    // Info
    state->noi = 1;
    state->nos = state->active_image->nos;

    // Methods
    state->method_image = std::vector<std::shared_ptr<Engine::Method>>( state->noi );
    state->method_chain = std::shared_ptr<Engine::Method>();

    // Set quiet method parameters
    if( state->quiet )
    {
        state->active_image->llg_parameters->output_any = false;
        state->active_image->mc_parameters->output_any  = false;
        state->chain->gneb_parameters->output_any       = false;
    }
    //------------------------------------------------------------------------------------------

    //---------------- Initial file writing (input, positions, neighbours) ---------------------
    Save_Initial_Final( state, true );
    //------------------------------------------------------------------------------------------

    //----------------------- Final log --------------------------------------------------------
    block.clear();
    auto now  = std::chrono::system_clock::now();
    auto diff = Timing::DateTimePassed( now - state->datetime_creation );
    block.emplace_back( "=====================================================" );
    block.emplace_back( "============ Spirit State: Initialised ==============" );
    block.emplace_back( "============     " + fmt::format( "NOS={} NOI={}", state->nos, state->noi ) );
    block.emplace_back( "    Initialisation took " + diff );
    block.emplace_back( "    Number of  Errors:  " + fmt::format( "{}", Log_Get_N_Errors( state ) ) );
    block.emplace_back( "    Number of Warnings: " + fmt::format( "{}", Log_Get_N_Warnings( state ) ) );
    block.emplace_back( "=====================================================" );
    Log( Log_Level::All, Log_Sender::All, block );
    // Try to write the log file
    try
    {
        Log.Append_to_File();
    }
    catch( ... )
    {
        spirit_handle_exception_api( -1, -1 );
    }
    //------------------------------------------------------------------------------------------

    return state;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return nullptr;
}

void State_Delete( State * state ) noexcept
try
{
    check_state( state );

    std::vector<std::string> block;
    block.emplace_back( "=====================================================" );
    block.emplace_back( "============ Spirit State: Deleting... ==============" );

    // Final file writing (input, positions, neighbours)
    Save_Initial_Final( state, false );

    // Timing
    auto now  = std::chrono::system_clock::now();
    auto diff = Timing::DateTimePassed( now - state->datetime_creation );
    block.emplace_back( fmt::format( "    State existed for {}", diff ) );
    block.emplace_back( fmt::format( "    Number of  Errors:  {}", Log_Get_N_Errors( state ) ) );
    block.emplace_back( fmt::format( "    Number of Warnings: {}", Log_Get_N_Warnings( state ) ) );

    // Delete
    delete( state );

    block.emplace_back( "============== Spirit State: Deleted ================" );
    block.emplace_back( "=====================================================" );

    Log( Log_Level::All, Log_Sender::All, block );
    Log.Append_to_File();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

void State_Update( State * state ) noexcept
try
{
    check_state( state );

    // Correct for removed images - active_image can maximally be noi-1
    if( state->chain->idx_active_image >= state->chain->noi )
        state->chain->idx_active_image = state->chain->noi - 1;

    // Update Image
    state->idx_active_image = state->chain->idx_active_image;
    state->active_image     = state->chain->images[state->idx_active_image];

    // Update NOS, NOI
    state->noi = state->chain->noi;
    state->nos = state->active_image->nos;
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

void State_To_Config( State * state, const char * config_file, const char * comment ) noexcept
try
{
    check_state( state );

    const std::string cfg = std::string( config_file );

    Log( Log_Level::Info, Log_Sender::All, "Writing State configuration to file " + cfg );

    if( std::filesystem::path( cfg ).extension() != ".toml" )
        Log( Log_Level::Warning, Log_Sender::All,
             "The output format is toml. Consider using a '.toml' file extension!" );

    IO::write_to_file( ::escape_lines( comment ), cfg );

    toml::table tbl;
    tbl.insert( "logging", IO::Logging_to_TOML() );
    tbl.insert( "geometry", IO::Geometry_to_TOML( state->active_image->hamiltonian->get_geometry() ) );
    tbl.insert(
        "method", toml::table{
                      { "llg", IO::Parameters_Method_LLG_to_TOML( *state->active_image->llg_parameters ) },
                      { "ema", IO::Parameters_Method_EMA_to_TOML( *state->active_image->ema_parameters ) },
                      { "mmf", IO::Parameters_Method_MMF_to_TOML( *state->active_image->mmf_parameters ) },
                      { "mc", IO::Parameters_Method_MC_to_TOML( *state->active_image->mc_parameters ) },
                      { "gneb", IO::Parameters_Method_GNEB_to_TOML( *state->chain->gneb_parameters ) },
                  } );
    tbl.insert( "hamiltonian", IO::Hamiltonian_to_TOML( *state->active_image->hamiltonian ) );

    IO::append_to_file( tbl, cfg );
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}

const char * State_DateTime( State * state ) noexcept
try
{
    check_state( state );
    return state->datetime_creation_string.c_str();
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
    return "00:00:00";
}

// Helper function for file writing at setup and delete of State.
//    Input, positions, neighbours.
void Save_Initial_Final( State * state, bool initial ) noexcept
try
{
    // Folder
    std::string folder = Log.output_folder;

    // Tag
    std::string tag = "";
    if( Log.file_tag == std::string( "<time>" ) )
        tag += state->datetime_creation_string + "_";
    else if( Log.file_tag != std::string( "" ) )
        tag += Log.file_tag + "_";

    // Suffix
    std::string suffix = "";
    if( initial )
        suffix += "initial";
    else
        suffix += "final";

    // Save the config
    try
    {
        if( ( Log.save_input_initial && initial ) || ( Log.save_input_final && !initial ) )
        {
            std::string file    = folder + "/input/" + tag + suffix + ".toml";
            std::string comment = fmt::format(
                "###\n### Original configuration file was called\n###   \"{}\"\n###\n", state->config_file );
            State_To_Config( state, file.c_str(), comment.c_str() );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( -1, -1 );
    }

    // Save the positions
    try
    {
        if( ( Log.save_positions_initial && initial ) || ( Log.save_positions_final && !initial ) )
        {
            std::string file = folder + "/output/" + tag + "positions_" + suffix + ".txt";
            IO_Positions_Write( state, file.c_str(), IO_Fileformat_OVF_text, state->config_file.c_str() );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( -1, -1 );
    }

    // Save the neighbours
    try
    {
        if( ( Log.save_neighbours_initial && initial ) || ( Log.save_neighbours_final && !initial ) )
        {
            std::string file = folder + "/output/" + tag + "neighbours_exchange_" + suffix + ".txt";
            IO_Image_Write_Neighbours_Exchange( state, file.c_str() );
            file = folder + "/output/" + tag + "neighbours_dmi_" + suffix + ".txt";
            IO_Image_Write_Neighbours_DMI( state, file.c_str() );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_api( -1, -1 );
    }
}
catch( ... )
{
    spirit_handle_exception_api( -1, -1 );
}
