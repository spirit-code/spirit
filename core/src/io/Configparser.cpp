#include <engine/Neighbours.hpp>
#include <engine/Vectormath.hpp>
#include <io/Configparser.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/IO.hpp>
#include <io/configparser/Converter.hpp>
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

auto Spin_System_from_TOML( const toml::table & tbl ) -> std::unique_ptr<::State::system_t>
try
{
    Log( Log_Level::Info, Log_Sender::IO, "-------------- Initialising Spin System ------------" );

    auto invoke = []( auto parser, auto * tbl, auto &&... args )
    { return std::invoke( parser, tbl ? *tbl : toml::table{}, std::forward<decltype( args )>( args )... ); };

    auto hamiltonian = invoke(
        Hamiltonian_from_TOML<::State::hamiltonian_t>, tbl["hamiltonian"].as_table(),
        /*geometry=*/invoke( Geometry_from_TOML, tbl["geometry"].as_table() ) );
    auto llg_params = invoke( Parameters_Method_LLG_from_TOML, tbl.at_path( "method.llg" ).as_table() );
    auto mc_params  = invoke( Parameters_Method_MC_from_TOML, tbl.at_path( "method.mc" ).as_table() );
    auto ema_params = invoke( Parameters_Method_EMA_from_TOML, tbl.at_path( "method.ema" ).as_table() );
    auto mmf_params = invoke( Parameters_Method_MMF_from_TOML, tbl.at_path( "method.mmf" ).as_table() );

    auto system = std::make_unique<::State::system_t>(
        std::move( hamiltonian ), std::move( llg_params ), std::move( mc_params ), std::move( ema_params ),
        std::move( mmf_params ), /*allow_iterations=*/false );

    Log( Log_Level::Info, Log_Sender::IO, "-------------- Spin System Initialised -------------" );

    return system;
}
catch( ... )
{
    spirit_handle_exception_core( fmt::format( "Unable to initialize spin system" ) );
    return nullptr;
} // End Spin_System_from_TOML

} // namespace IO
