#include <io/Filter_File_Handle.hpp>
#include <io/configparser/Converter.hpp>
#include <io/hamiltonian/Hamiltonian.hpp>
#include <io/hamiltonian/Interactions.hpp>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

#define log_error( expr )                                                                                              \
    if( auto error = ( expr ); error.has_value() )                                                                     \
        Log( Log_Level::Error, Log_Sender::IO, *error );

template<>
auto Hamiltonian_from_TOML( const toml::table & tbl, Data::Geometry geometry, intfield boundary_conditions )
    -> std::unique_ptr<Engine::Spin::Hamiltonian>
{
    namespace Interaction = Engine::Spin::Interaction;
    using Engine::Spin::Hamiltonian;

    std::vector<std::string> parameter_log;

    auto zeeman                    = Zeeman_from_TOML( tbl, parameter_log );
    auto [uniaxial_ani, cubic_ani] = Anisotropy_from_TOML( tbl, geometry, parameter_log );
    auto biaxial_ani               = Biaxial_Anisotropy_from_TOML( tbl, geometry, parameter_log );
    auto [exchange, dmi]           = Pair_Interactions_from_TOML( tbl, geometry, parameter_log );
    auto ddi                       = DDI_from_TOML( tbl, geometry, parameter_log );
    auto quadruplets               = Quadruplets_from_TOML( tbl, geometry, parameter_log );
    auto gaussian                  = Gaussian_from_TOML( tbl, parameter_log );

    Log( Log_Level::Debug, Log_Sender::IO, "Building Hamiltonian" );
    using Engine::Spin::Hamiltonian;
    auto hamiltonian = std::make_unique<Hamiltonian>( std::move( geometry ), std::move( boundary_conditions ) );
    zeeman.external_field_magnitude *= Utility::Constants::mu_B;
    log_error( hamiltonian->set_data<Interaction::Zeeman>( std::move( zeeman ) ) );
    log_error( hamiltonian->set_data<Interaction::Anisotropy>( std::move( uniaxial_ani ) ) );
    log_error( hamiltonian->set_data<Interaction::Cubic_Anisotropy>( std::move( cubic_ani ) ) );
    log_error( hamiltonian->set_data<Interaction::Biaxial_Anisotropy>( std::move( biaxial_ani ) ) );
    log_error( hamiltonian->set_data<Interaction::Exchange>( std::move( exchange ) ) );
    log_error( hamiltonian->set_data<Interaction::DMI>( std::move( dmi ) ) );
    log_error( hamiltonian->set_data<Interaction::DDI>( std::move( ddi ) ) );
    log_error( hamiltonian->set_data<Interaction::Quadruplet>( std::move( quadruplets ) ) );
    log_error( hamiltonian->set_data<Interaction::Gaussian>( std::move( gaussian ) ) );

    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Hamiltonian built: \"{}\"", hamiltonian->Name() ) );
    return hamiltonian;
}

template<>
auto Hamiltonian_from_Config(
    const std::string & config_file_name, Data::Geometry geometry,
    intfield boundary_conditions ) -> std::unique_ptr<Engine::Spin::Hamiltonian>
{
    return Hamiltonian_from_TOML<Engine::Spin::Hamiltonian>(
        convert::Hamiltonian( config_file_name ), std::move( geometry ), std::move( boundary_conditions ) );
}
} // namespace IO
