#include <io/Filter_File_Handle.hpp>
#include <io/Hamiltonian.hpp>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

template<>
auto Hamiltonian_from_TOML( const toml::table & root, Data::Geometry geometry )
    -> std::unique_ptr<Engine::Spin::Hamiltonian>
{
    namespace Interaction = Engine::Spin::Interaction;
    using Engine::Spin::Hamiltonian;
    const toml::table default_table{};
    const auto & tbl = [node_view = root.at_path( "hamiltonian" ).as_table(), &default_table]() -> const toml::table &
    {
        if( node_view )
            return *node_view;
        else
        {
            Log( Log_Level::Warning, Log_Sender::IO, "Missing config section: 'hamiltonian'. Using defaults..." );
            return default_table;
        }
    }();

    const auto log_error = []( std::optional<std::string> error ) -> void
    {
        if( error )
            Log( Log_Level::Error, Log_Sender::IO, *error );
    };

    std::vector<std::string> parameter_log{ "" };
    auto zeeman                    = Zeeman_from_TOML( tbl, parameter_log );
    auto [uniaxial_ani, cubic_ani] = Anisotropy_from_TOML( tbl, geometry, parameter_log );
    auto biaxial_ani               = Biaxial_Anisotropy_from_TOML( tbl, geometry, parameter_log );
    auto [exchange, dmi]           = Pair_Interactions_from_TOML( tbl, geometry, parameter_log );
    auto ddi                       = DDI_from_TOML( tbl, geometry, parameter_log );
    auto quadruplets               = Quadruplets_from_TOML( tbl, geometry, parameter_log );
    auto two_site_anisotropy       = Two_Site_Anisotropy_from_TOML( tbl, geometry, parameter_log );
    auto gaussian                  = Gaussian_from_TOML( tbl, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Building Hamiltonian" );
    using Engine::Spin::Hamiltonian;
    auto hamiltonian = std::make_unique<Hamiltonian>( std::move( geometry ) );
    log_error( hamiltonian->set_data<Interaction::Zeeman>( std::move( zeeman ) ) );
    log_error( hamiltonian->set_data<Interaction::Anisotropy>( std::move( uniaxial_ani ) ) );
    log_error( hamiltonian->set_data<Interaction::Cubic_Anisotropy>( std::move( cubic_ani ) ) );
    log_error( hamiltonian->set_data<Interaction::Biaxial_Anisotropy>( std::move( biaxial_ani ) ) );
    log_error( hamiltonian->set_data<Interaction::Exchange>( std::move( exchange ) ) );
    log_error( hamiltonian->set_data<Interaction::DMI>( std::move( dmi ) ) );
    log_error( hamiltonian->set_data<Interaction::DDI>( std::move( ddi ) ) );
    log_error( hamiltonian->set_data<Interaction::Quadruplet>( std::move( quadruplets ) ) );
    log_error( hamiltonian->set_data<Interaction::Two_Site_Anisotropy>( std::move( two_site_anisotropy ) ) );
    log_error( hamiltonian->set_data<Interaction::Gaussian>( std::move( gaussian ) ) );

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Hamiltonian built: \"{}\"", hamiltonian->Name() ) );
    parameter_log[0] = fmt::format( "Hamiltonian {}:", hamiltonian->Name() );
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );
    return hamiltonian;
}

auto Hamiltonian_to_TOML( const Engine::Spin::Hamiltonian & hamiltonian ) -> toml::table
{
    namespace Interaction = Engine::Spin::Interaction;
    toml::table tbl;
    const auto insert = [&tbl]( auto && values ) { tbl.insert( values.begin(), values.end() ); };

    insert( Zeeman_to_TOML( hamiltonian.data<Interaction::Zeeman>() ) );
    insert( Anisotropy_to_TOML(
        hamiltonian.data<Interaction::Anisotropy>(), hamiltonian.data<Interaction::Cubic_Anisotropy>() ) );
    insert( Pair_Interactions_to_TOML(
        hamiltonian.cache<Interaction::Exchange>(), hamiltonian.cache<Interaction::DMI>() ) );
    insert( Quadruplets_to_TOML( hamiltonian.data<Interaction::Quadruplet>() ) );
    insert( Gaussian_to_TOML( hamiltonian.data<Interaction::Gaussian>() ) );
    insert( Two_Site_Anisotropy_to_TOML( hamiltonian.data<Interaction::Two_Site_Anisotropy>() ) );
    insert( DDI_to_TOML( hamiltonian.data<Interaction::DDI>() ) );
    return tbl;
}

} // namespace IO
