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
                              bool normalize_by_nos )
	{
		bool readability_toggle = true;

		std::string separator = "";
		std::string line = "";
		for (unsigned int i=0; i<firstcolumns.size(); ++i)
		{
			if (readability_toggle) separator += "----------------------++";
			// Centered column titles
			line += fmt::format("{:^22}", firstcolumns[i]) + "||";
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
				line += fmt::format("{:^20}", pair.first);
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
                               const std::string filename, bool normalize_by_nos )
	{
		bool readability_toggle = true;
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
                              bool normalize_by_nos )
	{
		bool readability_toggle = true;
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
                                       bool normalize_by_nos )
	{
		bool readability_toggle = true;
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
                               const std::string filename, bool normalize_by_nos )
	{
		int isystem;
		bool readability_toggle = true;
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
                                            const std::string filename, bool normalize_by_nos )
	{
		int isystem, iinterp, idx;
		bool readability_toggle = true;
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
				std::string line = fmt::format("{:^20} || {:^20} || {:^20.10f} || {:^20.10f} ||", 
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


    void Write_Positions( const Data::Geometry& geometry, 
                          const std::string filename, VF_FileFormat format,
                          const std::string comment, bool append )
    {
        switch( format )
        {
            case VF_FileFormat::SPIRIT_WHITESPACE_SPIN:
            case VF_FileFormat::SPIRIT_WHITESPACE_POS_SPIN:
            case VF_FileFormat::SPIRIT_CSV_SPIN:
            case VF_FileFormat::SPIRIT_CSV_POS_SPIN:
                Write_SPIRIT_Version( filename, append );
                Save_To_SPIRIT( geometry.positions, geometry, filename, format, comment );
                break;
            case VF_FileFormat::OVF_BIN8:
            case VF_FileFormat::OVF_BIN4:
            case VF_FileFormat::OVF_TEXT:
            {
                oFile_OVF ofile_ovf( filename, format, comment );
                ofile_ovf.write_image( geometry.positions, geometry ); 
                break;
            }
            default:
                Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format( "Non "
                        "existent file format" ), -1, -1 );
                        
                // TODO: throw some exception to avoid logging "success" by API function
                        
                break;
        }        
    }


    void Write_Spin_Configuration( const vectorfield& vf, const Data::Geometry& geometry, 
                                   const std::string filename, VF_FileFormat format,
                                   const std::string comment, bool append )
    {
        switch( format )
        {
            case VF_FileFormat::SPIRIT_WHITESPACE_SPIN:
            case VF_FileFormat::SPIRIT_WHITESPACE_POS_SPIN:
            case VF_FileFormat::SPIRIT_CSV_SPIN:
            case VF_FileFormat::SPIRIT_CSV_POS_SPIN:
            Write_SPIRIT_Version( filename, append );
            Save_To_SPIRIT( vf, geometry, filename, format, comment );
            break;
            case VF_FileFormat::OVF_BIN8:
            case VF_FileFormat::OVF_BIN4:
            case VF_FileFormat::OVF_TEXT:
            {
                oFile_OVF ofile_ovf( filename, format, comment );
                ofile_ovf.write_image( vf, geometry ); 
                break;
            }
            default:
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format( "Non "
            "existent file format" ), -1, -1 );

            // TODO: throw some exception to avoid logging "success" by API function

            break;
        }        
    }

    void Write_Chain_Spin_Configuration( const std::shared_ptr<Data::Spin_System_Chain>& chain, 
                                         const std::string filename, VF_FileFormat format, 
                                         const std::string comment, bool append )
    {
        // except OVF format
        if ( format == VF_FileFormat::OVF_BIN8 || 
             format == VF_FileFormat::OVF_BIN4 ||
             format == VF_FileFormat::OVF_TEXT )
        {
            oFile_OVF ofile_ovf( filename, format, comment );
            ofile_ovf.write_chain( chain ); 
            return;
        }
        
        // write version
        Write_SPIRIT_Version( filename, append );
        
        // Header
        std::string output_to_file;
        output_to_file = fmt::format( "### Spin Chain Configuration for {} images with NOS = {} "
                                      "after iteration {}\n#\n", chain->noi, chain->images[0]->nos, 
                                      comment );
        Append_String_to_File( output_to_file, filename );
        
        for (int image = 0; image < chain->noi; ++image )
        {
            //// NOTE: with that implementation we are dumping the output_to_file twice for every
            // image. One for the image number header and one with the call to Save_To_SPIRIT(). 
            // Maybe this will add an overhead for large enough chains. To change that the arguments
            // of Save_To_SPIRIT() must be modified with a reference to output_to_file variable. So
            // that the buffer will be supplied by the caller. In that case many changes will must
            // be done in the code
            
            // Append the number of the image
            output_to_file = fmt::format( "# Image No {}\n", image );
            Append_String_to_File( output_to_file, filename );
            
            Save_To_SPIRIT( *chain->images[image]->spins, *chain->images[image]->geometry, 
                            filename, format, comment );
        }
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
                oFile_OVF ofile_ovf( filename, format, comment );
                ofile_ovf.write_eigenmodes( eigenvalues, modes, geometry ); 
                break;
            }
            default:
            Log( Utility::Log_Level::Error, Utility::Log_Sender::API, fmt::format( "Non "
            "existent file format" ), -1, -1 );
            
            // TODO: throw some exception to avoid logging "success" by API function
            
            break;
        }        
    }
    
    void Write_SPIRIT_Version( const std::string filename, bool append )
    {
        std::string output_to_file = fmt::format( "### SPIRIT Version {}\n", 
                                                  Utility::version_full );
        
        // If data have to be appended then append SPIRIT's version and then everything else is 
        // appended. If not the SPIRIT's version is dumped and then everything else is appended.
        if ( append )
            Append_String_to_File( output_to_file, filename );
        else
            Dump_to_File( output_to_file, filename );
    }
    
    void Save_To_SPIRIT( const vectorfield& vf, const Data::Geometry& geometry, 
                         const std::string filename, VF_FileFormat format, 
                         const std::string comment )
    {
        // Header
        std::string output_to_file = "";
        output_to_file.reserve(int(1E+08));
        output_to_file += fmt::format( "### Spin Configuration for NOS = {} comment: {}\n", 
                                       vf.size(), comment );
        
        // Delimiter
        std::string delimiter;
        if ( format == VF_FileFormat::SPIRIT_WHITESPACE_SPIN || 
             format == VF_FileFormat::SPIRIT_WHITESPACE_POS_SPIN )
        {
             delimiter = " ";
        }
        else if ( format == VF_FileFormat::SPIRIT_CSV_SPIN || 
                  format == VF_FileFormat::SPIRIT_CSV_POS_SPIN )
        {
            delimiter = ", ";
        }
        
        // Data
        if ( format == VF_FileFormat::SPIRIT_CSV_SPIN ||
             format == VF_FileFormat::SPIRIT_WHITESPACE_SPIN ) 
        {
            for (int iatom = 0; iatom < vf.size(); ++iatom)
            {
                #ifdef SPIRIT_ENABLE_DEFECTS
                if( geometry.atom_types[iatom] < 0 )
                    output_to_file += fmt::format( "{:20.10f}{}{:20.10f}{}{:20.10f}\n", 
                                                   0.0, delimiter, 0.0, delimiter, 0.0 );
                else
                #endif
                    output_to_file += fmt::format( "{:20.10f}{}{:20.10f}{}{:20.10f}\n", 
                                                    vf[iatom][0], delimiter, 
                                                    vf[iatom][1], delimiter, 
                                                    vf[iatom][2] );
            }
        }
        else if ( format == VF_FileFormat::SPIRIT_CSV_POS_SPIN || 
                  format == VF_FileFormat::SPIRIT_WHITESPACE_POS_SPIN )
        {
            for (int iatom = 0; iatom < vf.size(); ++iatom)
            {
                #ifdef SPIRIT_ENABLE_DEFECTS
                if( geometry.atom_types[iatom] < 0 )
                    output_to_file += fmt::format( "{:20.10f}{}{:20.10f}{}{:20.10f}{}"
                                                   "{:20.10f}{}{:20.10f}{}{:20.10f}\n",
                                                   geometry.positions[iatom][0], delimiter,
                                                   geometry.positions[iatom][1], delimiter,
                                                   geometry.positions[iatom][2], delimiter,
                                                   0.0, delimiter, 0.0, delimiter, 0.0 );
                else
                #endif
                    output_to_file += fmt::format( "{:20.10f}{}{:20.10f}{}{:20.10f}{}"
                                                   "{:20.10f}{}{:20.10f}{}{:20.10f}\n", 
                                                   geometry.positions[iatom][0], delimiter,
                                                   geometry.positions[iatom][1], delimiter,
                                                   geometry.positions[iatom][2], delimiter,
                                                   vf[iatom][0], delimiter, 
                                                   vf[iatom][1], delimiter, 
                                                   vf[iatom][2] );
            }
        }
        
        Append_String_to_File( output_to_file, filename );
    }

}
