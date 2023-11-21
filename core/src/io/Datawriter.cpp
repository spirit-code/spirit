#include <engine/Vectormath.hpp>
#include <io/Fileformat.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace IO
{

void Write_Neighbours_Exchange( const Data::Spin_System & system, const std::string & filename )
{
    pairfield exchange_pairs;
    scalarfield exchange_magnitudes;
    system.hamiltonian->getInteraction<Engine::Interaction::Exchange>()->getParameters(
        exchange_pairs, exchange_magnitudes );

    std::size_t n_neighbours = exchange_pairs.size();

#if defined( SPIRIT_USE_OPENMP )
    // When parallelising (cuda or openmp), all neighbours per spin are already there
    const bool mirror_neighbours = false;
#else
    // When running on a single thread, we need to re-create redundant neighbours
    const bool mirror_neighbours = true;
    n_neighbours *= 2;
#endif

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
            if( mirror_neighbours )
            {
                // Mirrored interactions
                output += fmt::format(
                    "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n", exchange_pairs[i].j, exchange_pairs[i].i,
                    ( -1 ) * exchange_pairs[i].translations[0], ( -1 ) * exchange_pairs[i].translations[1],
                    ( -1 ) * exchange_pairs[i].translations[2], exchange_magnitudes[i] );
            }
        }
    }

    dump_to_file( output, filename );
}

void Write_Neighbours_DMI( const Data::Spin_System & system, const std::string & filename )
{
    const auto & ham = system.hamiltonian;

    std::size_t n_neighbours = ham->dmi_pairs.size();

#if defined( SPIRIT_USE_OPENMP )
    // When parallelising (cuda or openmp), all neighbours per spin are already there
    const bool mirror_neighbours = false;
#else
    // When running on a single thread, we need to re-create redundant neighbours
    const bool mirror_neighbours = true;
    n_neighbours *= 2;
#endif

    std::string output;
    output.reserve( int( 0x02000000 ) ); // reserve 32[MByte]

    output += "###    Interaction neighbours:\n";
    output += fmt::format( "n_neighbours_dmi {}\n", n_neighbours );

    if( !ham->dmi_pairs.empty() )
    {
        output += fmt::format(
            "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15} {:^15} {:^15} {:^15}\n", "i", "j", "da", "db", "dc", "Dij",
            "Dijx", "Dijy", "Dijz" );
        for( std::size_t i = 0; i < ham->dmi_pairs.size(); ++i )
        {
            output += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n", ham->dmi_pairs[i].i,
                ham->dmi_pairs[i].j, ham->dmi_pairs[i].translations[0], ham->dmi_pairs[i].translations[1],
                ham->dmi_pairs[i].translations[2], ham->dmi_magnitudes[i], ham->dmi_normals[i][0],
                ham->dmi_normals[i][1], ham->dmi_normals[i][2] );
            if( mirror_neighbours )
            {
                // Mirrored interactions
                output += fmt::format(
                    "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                    ham->dmi_pairs[i].j, ham->dmi_pairs[i].i, ( -1 ) * ham->dmi_pairs[i].translations[0],
                    ( -1 ) * ham->dmi_pairs[i].translations[1], ( -1 ) * ham->dmi_pairs[i].translations[2],
                    ham->dmi_magnitudes[i], ( -1 ) * ham->dmi_normals[i][0], ( -1 ) * ham->dmi_normals[i][1],
                    ( -1 ) * ham->dmi_normals[i][2] );
            }
        }
    }

    dump_to_file( output, filename );
}

void Write_Energy_Header(
    const Data::Spin_System & system, const std::string & filename, const std::vector<std::string> && firstcolumns,
    bool contributions, bool normalize_by_nos, bool readability_toggle )
{
    std::string separator = "";
    std::string line      = "";
    for( const auto & column : firstcolumns )
    {
        if( readability_toggle )
            separator += "----------------------++";
        // Centered column titles
        line += fmt::format( " {:^20} ||", column );
    }
    if( contributions )
    {
        bool first = true;
        for( const auto & pair : system.E_array )
        {
            if( first )
                first = false;
            else
            {
                line += "|";
                if( readability_toggle )
                    separator += "+";
            }
            // Centered column titles
            line += fmt::format( " {:^20} ", pair.first );
            if( readability_toggle )
                separator += "----------------------";
        }
    }
    line += "\n";
    separator += "\n";

    std::string header;
    if( readability_toggle )
        header = separator + line + separator;
    else
        header = line;
    if( !readability_toggle )
        std::replace( header.begin(), header.end(), '|', ' ' );

    write_to_file( header, filename );
}

void Append_Image_Energy(
    const Data::Spin_System & system, const int iteration, const std::string & filename, bool normalize_by_nos,
    bool readability_toggle )
{
    scalar normalization = 1;
    if( normalize_by_nos )
        normalization = static_cast<scalar>( 1.0 / static_cast<double>( system.nos ) );

    // s.UpdateEnergy();

    // Centered column entries
    std::string line = fmt::format( " {:^20} || {:^20.10f} |", iteration, system.E * normalization );
    for( const auto & pair : system.E_array )
    {
        line += fmt::format( "| {:^20.10f} ", pair.second * normalization );
    }
    line += "\n";

    if( !readability_toggle )
        std::replace( line.begin(), line.end(), '|', ' ' );

    append_to_file( line, filename );
}

void Write_Image_Energy(
    const Data::Spin_System & system, const std::string & filename, bool normalize_by_nos, bool readability_toggle )
{
    scalar normalization = 1;
    if( normalize_by_nos )
        normalization = static_cast<scalar>( 1.0 / static_cast<double>( system.nos ) );

    Write_Energy_Header( system, filename, { "E_tot" } );

    std::string line = fmt::format( " {:^20.10f} |", system.E * normalization );
    for( const auto & pair : system.E_array )
    {
        line += fmt::format( "| {:^20.10f} ", pair.second * normalization );
    }
    line += "\n";

    if( !readability_toggle )
        std::replace( line.begin(), line.end(), '|', ' ' );

    append_to_file( line, filename );
}

void Write_Chain_Energies(
    const Data::Spin_System_Chain & chain, const int iteration, const std::string & filename, bool normalize_by_nos,
    bool readability_toggle )
{
    scalar normalization = 1;
    if( normalize_by_nos )
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

        if( !readability_toggle )
            std::replace( line.begin(), line.end(), '|', ' ' );

        append_to_file( line, filename );
    }
}

void Write_Chain_Energies_Interpolated(
    const Data::Spin_System_Chain & chain, const std::string & filename, bool normalize_by_nos,
    bool readability_toggle )
{
    scalar normalization = 1;
    if( normalize_by_nos )
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
            if( !readability_toggle )
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
