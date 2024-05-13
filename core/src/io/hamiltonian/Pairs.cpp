#include <engine/Vectormath_Defines.hpp>
#include <io/Dataparser.hpp>
#include <io/Filter_File_Handle.hpp>
#include <io/Tableparser.hpp>

#include <vector>

using Utility::Log_Level, Utility::Log_Sender;

namespace IO
{

namespace
{

// Read from Pairs file by Markus & Bernd
void Pairs_from_File(
    const std::string & pairs_file, const Data::Geometry & geometry, int & nop, pairfield & exchange_pairs,
    scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals ) noexcept
try
{
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Reading spin pairs from file \"{}\"", pairs_file ) );

    using PairTableParser
        = TableParser<int, int, int, int, int, scalar, scalar, scalar, scalar, scalar, scalar, scalar, scalar>;
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
                 fmt::format( "No interactions could be found in pairs file \"{}\"", pairs_file ) );

        return [DMI_xyz, DMI_abc, DMI_magnitude,
                &geometry]( const PairTableParser::read_row_t & row ) -> std::tuple<Pair, scalar, Vector3, scalar>
        {
            auto [i, j, da, db, dc, Dij, Dijx, Dijy, Dijz, Dija, Dijb, Dijc, Jij] = row;
            Vector3 D_temp;
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
    nop             = data.size();

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
             "Done reading {} spin pairs from file \"{}\", giving {} exchange and {} DM (symmetry-reduced) pairs.", nop,
             pairs_file, exchange_pairs.size(), dmi_pairs.size() ) );
}
catch( ... )
{
    spirit_rethrow( fmt::format( "Could not read pairs file \"{}\"", pairs_file ) );
}

} // namespace

void Pair_Interactions_from_Pairs_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    pairfield & exchange_pairs, scalarfield & exchange_magnitudes, pairfield & dmi_pairs, scalarfield & dmi_magnitudes,
    vectorfield & dmi_normals )
{
    std::string interaction_pairs_file{};
    int n_pairs = 0;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        // Interaction Pairs
        if( config_file_handle.Find( "n_interaction_pairs" ) )
            interaction_pairs_file = config_file_name;
        else if( config_file_handle.Find( "interaction_pairs_file" ) )
            config_file_handle >> interaction_pairs_file;

        if( !interaction_pairs_file.empty() )
        {
            // The file name should be valid so we try to read it
            Pairs_from_File(
                interaction_pairs_file, geometry, n_pairs, exchange_pairs, exchange_magnitudes, dmi_pairs,
                dmi_magnitudes, dmi_normals );
        }
        // else
        //{
        //	Log(Log_Level::Warning, Log_Sender::IO, "Hamiltonian_Heisenberg: Default Interaction pairs have not
        // been implemented yet."); 	throw Exception::System_not_Initialized;
        //	// Not implemented!
        //}
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to read interaction pairs from config file \"{}\"", config_file_name ) );
    }
}

void Pair_Interactions_from_Shells_from_Config(
    const std::string & config_file_name, const Data::Geometry & geometry, std::vector<std::string> & parameter_log,
    scalarfield & exchange_magnitudes, scalarfield & dmi_magnitudes, int & dm_chirality )
{
    std::string interaction_shells_file{};
    int n_shells_exchange = 0;
    int n_shells_dmi      = 0;

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        config_file_handle.Read_Single( n_shells_exchange, "n_shells_exchange" );
        if( exchange_magnitudes.size() != n_shells_exchange )
            exchange_magnitudes = scalarfield( n_shells_exchange );
        if( n_shells_exchange > 0 )
        {
            if( config_file_handle.Find( "jij" ) )
            {
                for( std::size_t ishell = 0; ishell < n_shells_exchange; ++ishell )
                    config_file_handle >> exchange_magnitudes[ishell];
            }
            else
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Hamiltonian_Heisenberg: Keyword 'jij' not found. Using Default: {}",
                         exchange_magnitudes[0] ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Failed to read exchange parameters from config file \"{}\"", config_file_name ) );
    }

    try
    {
        IO::Filter_File_Handle config_file_handle( config_file_name );

        config_file_handle.Read_Single( n_shells_dmi, "n_shells_dmi" );
        if( dmi_magnitudes.size() != n_shells_dmi )
            dmi_magnitudes = scalarfield( n_shells_dmi );
        if( n_shells_dmi > 0 )
        {
            if( config_file_handle.Find( "dij" ) )
            {
                for( unsigned int ishell = 0; ishell < n_shells_dmi; ++ishell )
                    config_file_handle >> dmi_magnitudes[ishell];
            }
            else
                Log( Log_Level::Warning, Log_Sender::IO,
                     fmt::format(
                         "Hamiltonian_Heisenberg: Keyword 'dij' not found. Using Default: {}", dmi_magnitudes[0] ) );
        }
        config_file_handle.Read_Single( dm_chirality, "dm_chirality" );
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Failed to read DMI parameters from config file \"{}\"", config_file_name ) );
    }

    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "n_shells_exchange", n_shells_exchange ) );
    if( n_shells_exchange > 0 )
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "J_ij[0]", exchange_magnitudes[0] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "n_shells_dmi", n_shells_dmi ) );
    if( n_shells_dmi > 0 )
        parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "D_ij[0]", dmi_magnitudes[0] ) );
    parameter_log.emplace_back( fmt::format( "    {:<21} = {}", "DM chirality", dm_chirality ) );
}

} // namespace IO
