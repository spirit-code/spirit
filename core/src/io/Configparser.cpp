#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <io/Configparser.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/hamiltonian/Hamiltonian.hpp>
#include <utility/Constants.hpp>
#include <utility/Exception.hpp>
#include <utility/Logging.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <toml++/toml.hpp>

#include <memory>
#include <string>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

std::unique_ptr<::State::system_t> Spin_System_from_Config( const std::string & config_file_name )
try
{
    Log( Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------" );

    // Geometry
    auto geometry = Geometry_from_Config( config_file_name );
    // Boundary conditions
    auto boundary_conditions = Boundary_Conditions_from_Config( config_file_name );
    // LLG Parameters
    auto llg_params = Parameters_Method_LLG_from_Config( config_file_name );
    // MC Parameters
    auto mc_params = Parameters_Method_MC_from_Config( config_file_name );
    // EMA Parameters
    auto ema_params = Parameters_Method_EMA_from_Config( config_file_name );
    // MMF Parameters
    auto mmf_params = Parameters_Method_MMF_from_Config( config_file_name );
    // Hamiltonian
    auto hamiltonian = Hamiltonian_from_Config<::State::hamiltonian_t>(
        config_file_name, std::move( geometry ), std::move( boundary_conditions ) );
    // Spin System
    auto system = std::make_unique<::State::system_t>(
        std::move( hamiltonian ), std::move( llg_params ), std::move( mc_params ), std::move( ema_params ),
        std::move( mmf_params ), false );

    Log( Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------" );

    return system;
}
catch( ... )
{
    spirit_handle_exception_core(
        fmt::format( "Unable to initialize spin system from config file \"{}\"", config_file_name ) );
    return nullptr;
} // End Spin_System_from_Config

} // namespace IO
