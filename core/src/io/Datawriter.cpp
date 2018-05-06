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

    void Write_Image_Energy_per_Spin( const Data::Spin_System & s, const std::string filename, 
                                      bool normalize_by_nos, bool readability_toggle )
    {
        scalar nd = 1.0; // nos divide
        if (normalize_by_nos) nd = 1.0 / s.nos;
        else nd = 1;

        // s.UpdateEnergy();

        Write_Energy_Header(s, filename, {"ispin", "E_tot"});
        
        std::vector<std::pair<std::string, scalarfield>> contributions_spins(0);
        s.hamiltonian->Energy_Contributions_per_Spin(*s.spins, contributions_spins);

        std::string data = "";
        for (int ispin=0; ispin<s.nos; ++ispin)
        {
            scalar E_spin=0;
            
            // BUG: if the energy is not updated at least one this will raise a SIGSEGV
            
            for (auto& contribution : contributions_spins) E_spin += contribution.second[ispin];
            data += fmt::format(" {:^20} || {:^20.10f} |", ispin, E_spin * nd);
            for (auto pair : contributions_spins)
            {
                data += fmt::format("| {:^20.10f} ", pair.second[ispin] * nd);
            }
            data += "\n";
        }

        if (!readability_toggle) std::replace( data.begin(), data.end(), '|', ' ');
        Append_String_to_File(data, filename);
    }

    void Write_System_Force(const Data::Spin_System & s, const std::string filename)
    {
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

    void Write_Chain_Energies_Interpolated( const Data::Spin_System_Chain & c, 
                                            const std::string filename, bool normalize_by_nos,
                                            bool readability_toggle )
    {
        int isystem, iinterp, idx;
        scalar nd = 1.0; // nos divide
        if (normalize_by_nos) nd = 1.0 / c.images[0]->nos;
        else nd = 1;

        Write_Energy_Header(*c.images[0], filename, {"image", "iinterp", "Rx", "E_tot"});

        for (isystem = 0; isystem < (int)c.noi; ++isystem)
        {
            auto& system = *c.images[isystem];

            for (iinterp = 0; iinterp < c.gneb_parameters->n_E_interpolations+1; ++iinterp)
            {
                idx = isystem * (c.gneb_parameters->n_E_interpolations+1) + iinterp;
                std::string line = fmt::format(" {:^20} || {:^20} || {:^20.10f} || {:^20.10f} ||", 
                                    isystem, iinterp, c.Rx_interpolated[idx], 
                                    c.E_interpolated[idx] * nd );
                
                // TODO: interpolated Energy contributions
                // bool first = true;
                // for (auto pair : system.E_array_interpolated)
                // {
                // 	if (first) first = false;
                // 	else
                // 	{
                // 		line += "|";;
                // 	}
                // 	line += center(pair.second * nd, 10, 20);
                // }
                line += "\n";

                if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
                Append_String_to_File(line, filename);

                // Exit the loop if we reached the end
                if (isystem == c.noi-1) break;
            }
        }
    }


    void Write_Chain_Forces(const Data::Spin_System_Chain & c, const std::string filename)
    {
        /////////////////
        // TODO: rewrite like save_energy functions
        /////////////////

        // //========================= Init local vars ================================
        // int isystem;
        // bool readability_toggle = true;
        // bool divide_by_nos = true;
        // scalar nd = 1.0; // nos divide
        // const int buffer_length = 200;
        // std::string output_to_file = "";
        // output_to_file.reserve(int(1E+08));
        // char buffer_string_conversion[buffer_length + 2];
        // snprintf(buffer_string_conversion, buffer_length, " isystem ||        |Force|        ||         F_max");
        // if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
        // output_to_file.append(buffer_string_conversion);
        // snprintf(buffer_string_conversion, buffer_length, "\n---------++----------------------++---------------------");
        // if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
        // //------------------------ End Init ----------------------------------------

        // for (isystem = 0; isystem < (int)c.images.size(); ++isystem) {
        // 	//c.images[isystem]->UpdateEnergy();
        // 	// TODO: Need image->UpdateForce() which can also be used for convergence tests
        // 	if (divide_by_nos) { nd = 1.0 / c.images[isystem]->nos; }
        // 	else { nd = 1; }
        // 	snprintf(buffer_string_conversion, buffer_length, "\n %6i  ||  %18.10f  ||  %18.10f",
        // 		isystem, 0.0 * nd, 0.0);
        // 	if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
        // 	output_to_file.append(buffer_string_conversion);
        // }
        // Dump_to_File(output_to_file, filename);
    }

    void Write_Eigenmodes( const std::vector<scalar>& eigenvalues, 
                           const std::vector<std::shared_ptr<vectorfield>>& modes, 
                           const Data::Geometry& geometry, const std::string filename, 
                           VF_FileFormat format, const std::string comment, bool append )
    {
        int n_modes = modes.size(); 
        switch( format )
        {
            case VF_FileFormat::SPIRIT_WHITESPACE_SPIN:
            case VF_FileFormat::SPIRIT_WHITESPACE_POS_SPIN:
            case VF_FileFormat::SPIRIT_CSV_SPIN:
            case VF_FileFormat::SPIRIT_CSV_POS_SPIN:
                Log( Utility::Log_Level::Error, Utility::Log_Sender::IO, fmt::format( "SPIRIT "
                    "Format does not support Chain write" ), -1, -1 );
            break;
            case VF_FileFormat::OVF_BIN8:
            case VF_FileFormat::OVF_BIN4:
            case VF_FileFormat::OVF_TEXT:
            {   
                File_OVF file_ovf( filename, format );
                file_ovf.write_eigenmodes( eigenvalues, modes, geometry ); 
                break;
            }
            default:
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format( "Non "
            "existent file format" ), -1, -1 );
            
            // TODO: throw some exception to avoid logging "success" by API function
            
            break;
        }        
    }
} 
