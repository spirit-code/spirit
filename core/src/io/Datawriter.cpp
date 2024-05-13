#include <engine/Vectormath.hpp>
#include <io/Datawriter.hpp>
#include <io/Fileformat.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace IO
{

void Write_Neighbours_Exchange( const State::system_t & system, const std::string & filename )
{
    const auto * cache = system.hamiltonian->cache<Engine::Spin::Interaction::Exchange>();
    if( cache == nullptr )
        return;

    const pairfield & exchange_pairs        = cache->pairs;
    const scalarfield & exchange_magnitudes = cache->magnitudes;

    const std::size_t n_neighbours = 2 * exchange_pairs.size();

    std::string output;
    output.reserve( int( 0x02000000 ) ); // reserve 32[MByte]

    output += "###    Interaction neighbours:\n";
    output += fmt::format( "n_neighbours_exchange {}\n", n_neighbours );

    if( !exchange_pairs.empty() )
    {
        output += fmt::format( "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}\n", "i", "j", "da", "db", "dc", "Jij" );
        for( std::size_t i = 0; i < exchange_pairs.size(); ++i )
        {
            output += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n", exchange_pairs[i].i, exchange_pairs[i].j,
                exchange_pairs[i].translations[0], exchange_pairs[i].translations[1], exchange_pairs[i].translations[2],
                exchange_magnitudes[i] );
            // Mirrored interactions
            output += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n", exchange_pairs[i].j, exchange_pairs[i].i,
                ( -1 ) * exchange_pairs[i].translations[0], ( -1 ) * exchange_pairs[i].translations[1],
                ( -1 ) * exchange_pairs[i].translations[2], exchange_magnitudes[i] );
        }
    }

    dump_to_file( output, filename );
}

void Write_Neighbours_DMI( const State::system_t & system, const std::string & filename )
{
    const auto * cache = system.hamiltonian->cache<Engine::Spin::Interaction::DMI>();
    if( cache == nullptr )
        return;

    const pairfield & dmi_pairs        = cache->pairs;
    const scalarfield & dmi_magnitudes = cache->magnitudes;
    const vectorfield & dmi_normals    = cache->normals;

    const std::size_t n_neighbours = 2 * dmi_pairs.size();

    std::string output;
    output.reserve( int( 0x02000000 ) ); // reserve 32[MByte]

    output += "###    Interaction neighbours:\n";
    output += fmt::format( "n_neighbours_dmi {}\n", n_neighbours );

    if( !dmi_pairs.empty() )
    {
        output += fmt::format(
            "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15} {:^15} {:^15} {:^15}\n", "i", "j", "da", "db", "dc", "Dij",
            "Dijx", "Dijy", "Dijz" );
        for( std::size_t i = 0; i < dmi_pairs.size(); ++i )
        {
            output += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n", dmi_pairs[i].i,
                dmi_pairs[i].j, dmi_pairs[i].translations[0], dmi_pairs[i].translations[1],
                dmi_pairs[i].translations[2], dmi_magnitudes[i], dmi_normals[i][0], dmi_normals[i][1],
                dmi_normals[i][2] );
            // Mirrored interactions
            output += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n", dmi_pairs[i].j,
                dmi_pairs[i].i, ( -1 ) * dmi_pairs[i].translations[0], ( -1 ) * dmi_pairs[i].translations[1],
                ( -1 ) * dmi_pairs[i].translations[2], dmi_magnitudes[i], ( -1 ) * dmi_normals[i][0],
                ( -1 ) * dmi_normals[i][1], ( -1 ) * dmi_normals[i][2] );
        }
    }

    dump_to_file( output, filename );
}

void Write_Energy_Header(
    const State::system_t & system, const std::string & filename, const std::vector<std::string> && firstcolumns,
    Flags flags )
{
    verify_flags( flags, Flag::Contributions | Flag::Readability | Flag::Normalize_by_nos, __func__ );

    std::string separator = "";
    std::string line      = "";
    for( const auto & column : firstcolumns )
    {
        if( flags & Flag::Readability )
            separator += "----------------------++";
        // Centered column titles
        line += fmt::format( " {:^20} ||", column );
    }
    if( flags & Flag::Contributions )
    {
        bool first = true;
        for( const auto & pair : system.E_array )
        {
            if( first )
                first = false;
            else
            {
                line += "|";
                if( flags & Flag::Readability )
                    separator += "+";
            }
            // Centered column titles
            line += fmt::format( " {:^20} ", pair.first );
            if( flags & Flag::Readability )
                separator += "----------------------";
        }
    }
    line += "\n";
    separator += "\n";

    std::string header;
    if( flags & Flag::Readability )
        header = separator + line + separator;
    else
    {
        header = line;
        std::replace( header.begin(), header.end(), '|', ' ' );
    }

    write_to_file( header, filename );
}

void Append_Image_Energy(
    const State::system_t & system, const int iteration, const std::string & filename, Flags flags )
{
    verify_flags( flags, Flag::Readability | Flag::Normalize_by_nos, __func__ );

    scalar normalization = 1;
    if( flags & Flag::Normalize_by_nos )
        normalization = static_cast<scalar>( 1.0 / static_cast<double>( system.nos ) );

    // s.UpdateEnergy();

    // Centered column entries
    std::string line = fmt::format( " {:^20} || {:^20.10f} |", iteration, system.E * normalization );
    for( const auto & pair : system.E_array )
    {
        line += fmt::format( "| {:^20.10f} ", pair.second * normalization );
    }
    line += "\n";

    if( !( flags & Flag::Readability ) )
        std::replace( line.begin(), line.end(), '|', ' ' );

    append_to_file( line, filename );
}

void Write_Image_Energy( const State::system_t & system, const std::string & filename, Flags flags )
{
    verify_flags( flags, Flag::Readability | Flag::Normalize_by_nos, __func__ );

    scalar normalization = 1;
    if( flags & Flag::Normalize_by_nos )
        normalization = static_cast<scalar>( 1.0 / static_cast<double>( system.nos ) );

    Write_Energy_Header( system, filename, { "E_tot" } );

    std::string line = fmt::format( " {:^20.10f} |", system.E * normalization );
    for( const auto & pair : system.E_array )
    {
        line += fmt::format( "| {:^20.10f} ", pair.second * normalization );
    }
    line += "\n";

    if( !( flags & Flag::Readability ) )
        std::replace( line.begin(), line.end(), '|', ' ' );

    append_to_file( line, filename );
}

void Write_Chain_Energies(
    const State::chain_t & chain, const int iteration, const std::string & filename, Flags flags )
{
    verify_flags( flags, Flag::Readability | Flag::Normalize_by_nos, __func__ );

    scalar normalization = 1;
    if( flags & Flag::Normalize_by_nos )
        normalization = static_cast<scalar>( 1.0 / static_cast<double>( chain.images[0]->nos ) );

    Write_Energy_Header( *chain.images[0], filename, { "image", "Rx", "E_tot" } );

    for( int isystem = 0; isystem < chain.noi; ++isystem )
    {
        auto & system    = *chain.images[isystem];
        std::string line = fmt::format(
            " {:^20} || {:^20.10f} || {:^20.10f} |", isystem, chain.Rx[isystem], system.E * normalization );
        for( const auto & pair : system.E_array )
        {
            line += fmt::format( "| {:^20.10f} ", pair.second * normalization );
        }
        line += "\n";

        if( !( flags & Flag::Readability ) )
            std::replace( line.begin(), line.end(), '|', ' ' );

        append_to_file( line, filename );
    }
}

void Write_Chain_Energies_Interpolated( const State::chain_t & chain, const std::string & filename, Flags flags )
{
    verify_flags( flags, Flag::Readability | Flag::Normalize_by_nos, __func__ );

    scalar normalization = 1;
    if( flags & Flag::Normalize_by_nos )
        normalization = static_cast<scalar>( 1.0 / static_cast<double>( chain.images[0]->nos ) );

    Write_Energy_Header( *chain.images[0], filename, { "image", "iinterp", "Rx", "E_tot" } );

    for( int isystem = 0; isystem < chain.noi; ++isystem )
    {
        auto & system = *chain.images[isystem];

        for( int iinterp = 0; iinterp < chain.gneb_parameters->n_E_interpolations + 1; ++iinterp )
        {
            int idx          = isystem * ( chain.gneb_parameters->n_E_interpolations + 1 ) + iinterp;
            std::string line = fmt::format(
                " {:^20} || {:^20} || {:^20.10f} || {:^20.10f} ||", isystem, iinterp, chain.Rx_interpolated[idx],
                chain.E_interpolated[idx] * normalization );

            // TODO: interpolated Energy contributions
            bool first = true;
            for( std::size_t p = 0; p < system.E_array.size(); p++ )
            {
                if( first )
                    first = false;
                else
                    line += "|";

                line += fmt::format( " {:^20.10f} ", chain.E_array_interpolated[p][idx] * normalization );
            }
            line += "\n";

            // Whether to use space or | as column separator
            if( !( flags & Flag::Readability ) )
                std::replace( line.begin(), line.end(), '|', ' ' );

            // Write
            append_to_file( line, filename );

            // Exit the loop if we reached the end
            if( isystem == chain.noi - 1 )
                break;
        }
    }
}

} // namespace IO
