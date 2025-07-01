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

namespace
{

auto Boundary_Conditions_from_TOML( const toml::table & tbl ) -> intfield
{
    intfield boundary_conditions{ false, false, false };
    if( auto array = tbl["boundary_conditions"].as_array() )
    {
        const int array_size = static_cast<int>( array->size() );
        if( array->size() != 3 )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format(
                     "Too {} entries in boundary conditions, expected 3, found {}", array_size < 3 ? "few" : "many",
                     array_size ) );

        for( int i = 0; i < 3 && i < array_size; ++i )
        {
            if( auto value = ( *array )[i].as_integer() )
                boundary_conditions[i] = static_cast<int>( *value != 0 );
            else
                spirit_throw(
                    Utility::Exception_Classifier::Input_parse_failed, Log_Level::Error,
                    "Invalid type in boundary conditions encountered." );
        }
    }
    else
        Log( Log_Level::Warning, Log_Sender::IO,
             "No boundary conditions specified, using open boundary conditions in all "
             "dimensions." );
    return boundary_conditions;
}

auto Boundary_Conditions_to_TOML( const intfield & bc ) -> toml::table
{
    return toml::table{ { "boundary_conditions", toml_array_from_container( bc ) } };
}

} // namespace

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

    auto boundary_conditions = Boundary_Conditions_from_TOML( tbl );

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
    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );
    return hamiltonian;
}

auto Hamiltonian_to_TOML( const Engine::Spin::Hamiltonian & hamiltonian ) -> toml::table
{
    namespace Interaction = Engine::Spin::Interaction;
    const auto insert     = []( auto & tbl, auto && values ) { tbl.insert( values.begin(), values.end() ); };

    toml::table tbl;
    insert( tbl, Boundary_Conditions_to_TOML( hamiltonian.get_boundary_conditions() ) );
    insert( tbl, Zeeman_to_TOML( hamiltonian.data<Interaction::Zeeman>() ) );
    insert(
        tbl, Anisotropy_to_TOML(
                 hamiltonian.data<Interaction::Anisotropy>(), hamiltonian.data<Interaction::Cubic_Anisotropy>() ) );
    insert(
        tbl, Pair_Interactions_to_TOML(
                 hamiltonian.cache<Interaction::Exchange>(), hamiltonian.cache<Interaction::DMI>() ) );
    insert( tbl, Quadruplets_to_TOML( hamiltonian.data<Interaction::Quadruplet>() ) );
    insert( tbl, Gaussian_to_TOML( hamiltonian.data<Interaction::Gaussian>() ) );
    insert( tbl, DDI_to_TOML( hamiltonian.data<Interaction::DDI>() ) );
    return tbl;
}

} // namespace IO
