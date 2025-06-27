#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>
#include <io/configparser/Converter.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace
{

// Read from Pairs file by Markus & Bernd
void Pairs_from_File(
    Filter_File_Handle & pairs_file, const Data::Geometry & geometry, pairfield & exchange_pairs,
    scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading spin pairs from \"{}\"", pairs_file.filename() ) );

    using PairTableParser = TableParserInit<std::array<int, 5>, std::array<scalar, 8>>;
    const PairTableParser parser(
        { "i", "j", "da", "db", "dc", "dij", "dijx", "dijy", "dijz", "dija", "dijb", "dijc", "jij" } );

    auto transform_factory = [&pairs_file, &geometry]( const std::map<std::string_view, int> & idx )
    {
        bool DMI_xyz = false, DMI_abc = false, DMI_magnitude = false;

        if( idx.at( "dijx" ) >= 0 && idx.at( "dijy" ) >= 0 && idx.at( "dijz" ) >= 0 )
            DMI_xyz = true;
        if( idx.at( "dija" ) >= 0 && idx.at( "dijb" ) >= 0 && idx.at( "dijc" ) >= 0 )
            DMI_abc = true;
        if( idx.at( "dij" ) >= 0 )
            DMI_magnitude = true;

        if( idx.at( "j" ) < 0 && !DMI_xyz && !DMI_abc )
            Log( Log_Level::Warning, Log_Sender::IO,
                 fmt::format( "No interactions could be found in pairs file \"{}\"", pairs_file.filename() ) );

        return [DMI_xyz, DMI_abc, DMI_magnitude,
                &geometry]( const PairTableParser::read_row_t & row ) -> std::tuple<Pair, scalar, Vector3, scalar>
        {
            auto [i, j, da, db, dc, Dij, Dijx, Dijy, Dijz, Dija, Dijb, Dijc, Jij] = row;

            Vector3 D_temp = Vector3::Zero();
            if( DMI_xyz )
                D_temp = { Dijx, Dijy, Dijz };
            // Anisotropy vector orientation
            if( DMI_abc )
            {
                D_temp = { Dija, Dijb, Dijc };
                D_temp = { D_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[0] ),
                           D_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[1] ),
                           D_temp.dot( geometry.lattice_constant * geometry.bravais_vectors[2] ) };
            }

            if( !DMI_magnitude )
                Dij = D_temp.norm();

            D_temp.normalize();

            return std::make_tuple( Pair{ i, j, { da, db, dc } }, Jij, D_temp, Dij );
        };
    };

    const auto data = parser.parse( pairs_file, "n_interaction_pairs", 20, transform_factory );

    {
        auto predicate = []( const auto & first, const auto & second ) -> int
        {
            const auto t1 = std::array{ first.translations[0], first.translations[1], first.translations[2] };
            const auto t2 = std::array{ second.translations[0], second.translations[1], second.translations[2] };

            if( first.i == second.i && first.j == second.j && t1 == std::array{ t2[0], t2[1], t2[2] } )
                return 1;
            else if( first.i == second.j && first.j == second.i && t1 == std::array{ -t2[0], -t2[1], -t2[2] } )
                return -1;
            else
                return 0;
        };

        const auto reset = [size = data.size()]( auto & container ) { container.clear(), container.reserve( size ); };
        reset( exchange_pairs );
        reset( exchange_magnitudes );
        reset( dmi_pairs );
        reset( dmi_normals );
        reset( dmi_magnitudes );

        // Add the indices and parameters to the corresponding lists and deduplicate entries
        for( const auto & [pair, Jij, D_vec, Dij] : data )
        {
            if( Jij != 0 )
            {
                bool already_in{ false };
                int atposition = -1;
                for( std::size_t icheck = 0; icheck < exchange_pairs.size(); ++icheck )
                {
                    if( predicate( pair, exchange_pairs[icheck] ) == 0 )
                        continue;

                    already_in = true;
                    atposition = icheck;
                    break;
                }
                if( already_in )
                {
                    exchange_magnitudes[atposition] += Jij;
                }
                else
                {
                    exchange_pairs.push_back( pair );
                    exchange_magnitudes.push_back( Jij );
                }
            }
            if( Dij != 0 )
            {
                bool already_in{ false };
                int dfact      = 1;
                int atposition = -1;
                for( std::size_t icheck = 0; icheck < dmi_pairs.size(); ++icheck )
                {
                    const auto pred = predicate( pair, dmi_pairs[icheck] );
                    if( pred == 0 )
                        continue;

                    already_in = true;
                    atposition = icheck;
                    dfact      = pred;
                    break;
                }
                if( already_in )
                {
                    // Calculate new D vector by adding the two redundant ones and normalize again
                    Vector3 newD    = dmi_magnitudes[atposition] * dmi_normals[atposition] + dfact * Dij * D_vec;
                    scalar newdnorm = newD.norm();
                    newD.normalize();
                    dmi_magnitudes[atposition] = newdnorm;
                    dmi_normals[atposition]    = newD;
                }
                else
                {
                    dmi_pairs.push_back( pair );
                    dmi_magnitudes.push_back( Dij );
                    dmi_normals.push_back( D_vec );
                }
            }
        }
    }

    Log( Log_Level::Parameter, Log_Sender::IO,
         fmt::format(
             "Done reading {} spin pairs from \"{}\", giving {} exchange and {} DM (symmetry-reduced) pairs.",
             data.size(), pairs_file.filename(), exchange_pairs.size(), dmi_pairs.size() ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pairs file \"{}\"", pairs_file.filename() ) );
}

} // namespace

void Pair_Interactions_from_Pairs_from_TOML(
    const toml::table & tbl, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    pairfield & exchange_pairs, scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals )
{
    auto pairs = tbl["pairs"].as_string();
    if( !pairs )
        return;

    auto pairs_handle = Filter_File_Handle::from_string( pairs->get() );
    Pairs_from_File(
        pairs_handle, geometry, exchange_pairs, exchange_magnitudes, dmi_pairs, dmi_magnitudes, dmi_normals );
}

void Pair_Interactions_from_Shells_from_TOML(
    const toml::table & tbl, const Data::Geometry &, std::vector<std::string> & parameter_log,
    scalarfield & exchange_magnitudes, scalarfield & dmi_magnitudes, int & dm_chirality )
{
    if( auto exchange_shells = tbl["exchange_shells"].as_array() )
        exchange_magnitudes = toml_array_transform<scalarfield>::transform( *exchange_shells );

    if( auto dmi_shells = tbl["dmi_shells"].as_array() )
        dmi_magnitudes = toml_array_transform<scalarfield>::transform( *dmi_shells );

    dm_chirality = tbl["dmi_chirality"].value_or<int>( 0 );

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "n_shells_exchange", exchange_magnitudes.size() ) );
    if( !exchange_magnitudes.empty() )
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "J_ij[0]", exchange_magnitudes[0] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "n_shells_dmi", dmi_magnitudes.size() ) );
    if( !dmi_magnitudes.empty() )
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "D_ij[0]", dmi_magnitudes[0] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "DM chirality", dm_chirality ) );
}

void Pair_Interactions_from_Pairs_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    pairfield & exchange_pairs, scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals )
{

    return Pair_Interactions_from_Pairs_from_TOML(
        convert::Interaction::Pair_Interactions_from_Pairs( config_file_name ), geometry, parameter_log, exchange_pairs,
        exchange_magnitudes, dmi_pairs, dmi_magnitudes, dmi_normals );
}

void Pair_Interactions_from_Shells_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    scalarfield & exchange_magnitudes, scalarfield & dmi_magnitudes, int & dm_chirality )
{
    return Pair_Interactions_from_Shells_from_TOML(
        convert::Interaction::Pair_Interactions_from_Shells( config_file_name ), geometry, parameter_log,
        exchange_magnitudes, dmi_magnitudes, dm_chirality );
}

} // namespace IO
