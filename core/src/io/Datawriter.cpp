#include <io/IO.hpp>
#include <io/Fileformat.hpp>
#include <io/OVF_File.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cctype>

#include <fmt/format.h>

#ifdef CORE_USE_THREADS
#include <thread>
#endif

namespace IO
{
    void Write_Neighbours_Exchange( const Data::Spin_System& system, const std::string filename )
    {
        Engine::Hamiltonian_Heisenberg* ham =
            (Engine::Hamiltonian_Heisenberg *) system.hamiltonian.get();
        int n_neighbours = ham->exchange_pairs.size();

        #if defined(SPIRIT_USE_OPENMP)
        // When parallelising (cuda or openmp), all neighbours per spin are already there
        const bool mirror_neighbours = false;
        #else
        // When running on a single thread, we need to re-create redundant neighbours
        const bool mirror_neighbours = true;
        n_neighbours *= 2;
        #endif

        std::string output;
        output.reserve( int( 0x02000000 ) );  // reserve 32[MByte]

        output += "###    Interaction neighbours:\n";
        output += fmt::format( "n_neighbours_exchange {}\n", n_neighbours );

        if (ham->exchange_pairs.size() > 0)
        {
            output += fmt::format( "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15}\n",
                "i", "j", "da", "db", "dc", "Jij" );
            for (unsigned int i=0; i<ham->exchange_pairs.size(); ++i)
            {
                output += fmt::format( "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n",
                    ham->exchange_pairs[i].i, ham->exchange_pairs[i].j,
                    ham->exchange_pairs[i].translations[0], ham->exchange_pairs[i].translations[1],
                    ham->exchange_pairs[i].translations[2], ham->exchange_magnitudes[i] );
                if( mirror_neighbours )
                {
                    // Mirrored interactions
                    output += fmt::format( "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f}\n",
                        ham->exchange_pairs[i].j, ham->exchange_pairs[i].i,
                        (-1) * ham->exchange_pairs[i].translations[0],
                        (-1) * ham->exchange_pairs[i].translations[1],
                        (-1) * ham->exchange_pairs[i].translations[2],
                        ham->exchange_magnitudes[i] );
                }
            }
        }

        Dump_to_File( output, filename );
    }

    void Write_Neighbours_DMI( const Data::Spin_System& system, const std::string filename )
    {
        Engine::Hamiltonian_Heisenberg* ham =
            (Engine::Hamiltonian_Heisenberg *) system.hamiltonian.get();
        int n_neighbours = ham->dmi_pairs.size();

        #if defined(SPIRIT_USE_OPENMP)
        // When parallelising (cuda or openmp), all neighbours per spin are already there
        const bool mirror_neighbours = false;
        #else
        // When running on a single thread, we need to re-create redundant neighbours
        const bool mirror_neighbours = true;
        n_neighbours *= 2;
        #endif

        std::string output;
        output.reserve( int( 0x02000000 ) );  // reserve 32[MByte]

        output += "###    Interaction neighbours:\n";
        output += fmt::format( "n_neighbours_dmi {}\n", n_neighbours );

        if (ham->dmi_pairs.size() > 0)
        {
            output += fmt::format(
                "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15} {:^15} {:^15} {:^15}\n",
                "i", "j", "da", "db", "dc", "Dij", "Dijx", "Dijy", "Dijz");
            for (unsigned int i = 0; i<ham->dmi_pairs.size(); ++i)
            {
                output += fmt::format(
                    "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                    ham->dmi_pairs[i].i, ham->dmi_pairs[i].j,
                    ham->dmi_pairs[i].translations[0], ham->dmi_pairs[i].translations[1],
                    ham->dmi_pairs[i].translations[2], ham->dmi_magnitudes[i],
                    ham->dmi_normals[i][0], ham->dmi_normals[i][1], ham->dmi_normals[i][2]);
                if( mirror_neighbours )
                {
                    // Mirrored interactions
                    output += fmt::format(
                        "{:^3} {:^3}    {:^3} {:^3} {:^3}    {:^15.8f} {:^15.8f} {:^15.8f} {:^15.8f}\n",
                        ham->dmi_pairs[i].j, ham->dmi_pairs[i].i, (-1) * ham->dmi_pairs[i].translations[0],
                        (-1) * ham->dmi_pairs[i].translations[1], (-1) * ham->dmi_pairs[i].translations[2],
                        ham->dmi_magnitudes[i], (-1) * ham->dmi_normals[i][0],
                        (-1) * ham->dmi_normals[i][1], (-1) * ham->dmi_normals[i][2]);
                }
            }
        }

        Dump_to_File( output, filename );
    }

    void Write_Energy_Header( const Data::Spin_System & s, const std::string filename,
                                std::vector<std::string> firstcolumns, bool contributions,
                                bool normalize_by_nos, bool readability_toggle )
    {
        std::string separator = "";
        std::string line = "";
        for (unsigned int i=0; i<firstcolumns.size(); ++i)
        {
            if (readability_toggle) separator += "----------------------++";
            // Centered column titles
            line += fmt::format(" {:^20} ||", firstcolumns[i]);
        }
        if (contributions)
        {
            bool first = true;
            for (auto pair : s.E_array)
            {
                if (first) first = false;
                else
                {
                    line += "|";;
                    if (readability_toggle) separator += "+";
                }
                // Centered column titles
                line += fmt::format(" {:^20} ", pair.first);
                if (readability_toggle) separator += "----------------------";
            }
        }
        line += "\n";
        separator += "\n";

        std::string header;
        if (readability_toggle) header = separator + line + separator;
        else header = line;
        if (!readability_toggle) std::replace( header.begin(), header.end(), '|', ' ');
        String_to_File(header, filename);
    }

    void Append_Image_Energy( const Data::Spin_System & s, const int iteration,
                              const std::string filename, bool normalize_by_nos,
                              bool readability_toggle )
    {
        scalar nd = 1.0; // nos divide
        if (normalize_by_nos) nd = 1.0 / s.nos;
        else nd = 1;

        // s.UpdateEnergy();

        // Centered column entries
        std::string line = fmt::format(" {:^20} || {:^20.10f} |", iteration, s.E * nd);
        for (auto pair : s.E_array)
        {
            line += fmt::format("| {:^20.10f} ", pair.second * nd);
        }
        line += "\n";

        if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
        Append_String_to_File(line, filename);
    }

    void Write_Image_Energy( const Data::Spin_System & system, const std::string filename,
                             bool normalize_by_nos, bool readability_toggle )
    {
        scalar nd = 1.0; // nos divide
        if (normalize_by_nos) nd = 1.0 / system.nos;
        else nd = 1;

        Write_Energy_Header(system, filename, {"E_tot"});

        std::string line = fmt::format(" {:^20.10f} |", system.E * nd);
        for (auto pair : system.E_array)
        {
            line += fmt::format("| {:^20.10f} ", pair.second * nd);
        }
        line += "\n";

        if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
        Append_String_to_File(line, filename);
    }

    void Write_Chain_Energies( const Data::Spin_System_Chain & c, const int iteration,
                               const std::string filename, bool normalize_by_nos,
                               bool readability_toggle )
    {
        int isystem;
        scalar nd = 1.0; // nos divide
        if (normalize_by_nos) nd = 1.0 / c.images[0]->nos;
        else nd = 1;

        Write_Energy_Header(*c.images[0], filename, {"image", "Rx", "E_tot"});

        for (isystem = 0; isystem < (int)c.noi; ++isystem)
        {
            auto& system = *c.images[isystem];
            std::string line = fmt::format(" {:^20} || {:^20.10f} || {:^20.10f} |", isystem,
                                c.Rx[isystem], system.E * nd );
            for (auto pair : system.E_array)
            {
                line += fmt::format("| {:^20.10f} ", pair.second * nd);
            }
            line += "\n";

            if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
            Append_String_to_File(line, filename);
        }
    }

    void Write_Chain_Energies_Interpolated( const Data::Spin_System_Chain & chain,
                                            const std::string filename, bool normalize_by_nos,
                                            bool readability_toggle )
    {
        int isystem, iinterp, idx;
        scalar nd = 1.0; // nos divide
        if (normalize_by_nos) nd = 1.0 / chain.images[0]->nos;
        else nd = 1;

        Write_Energy_Header(*chain.images[0], filename, {"image", "iinterp", "Rx", "E_tot"});

        for (isystem = 0; isystem < (int)chain.noi; ++isystem)
        {
            auto& system = *chain.images[isystem];

            for (iinterp = 0; iinterp < chain.gneb_parameters->n_E_interpolations+1; ++iinterp)
            {
                idx = isystem * (chain.gneb_parameters->n_E_interpolations+1) + iinterp;
                std::string line = fmt::format(" {:^20} || {:^20} || {:^20.10f} || {:^20.10f} ||",
                                    isystem, iinterp, chain.Rx_interpolated[idx],
                                    chain.E_interpolated[idx] * nd );

                // TODO: interpolated Energy contributions
                bool first = true;
                for (auto pair : system.E_array)
                {
                    if (first)
                        first = false;
                    else
                        line += "|";

                    line += fmt::format(" {:^20.10f} ", 0.0);
                }
                line += "\n";

                // Whether to use space or | as column separator
                if( !readability_toggle )
                    std::replace( line.begin(), line.end(), '|', ' ');

                // Write
                Append_String_to_File(line, filename);

                // Exit the loop if we reached the end
                if( isystem == chain.noi-1 )
                    break;
            }
        }
    }
}