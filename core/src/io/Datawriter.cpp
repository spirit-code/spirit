#include <io/IO.hpp>
#include <engine/Vectormath.hpp>
#include <utility/Logging.hpp>

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
	void Write_Energy_Header(const Data::Spin_System & s, const std::string fileName, std::vector<std::string> firstcolumns, bool contributions, bool normalize_by_nos)
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
		String_to_File(header, fileName);
	}

	void Append_System_Energy(const Data::Spin_System & s, const int iteration, const std::string fileName, bool normalize_by_nos)
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
		Append_String_to_File(line, fileName);
	}

	void Write_System_Energy(const Data::Spin_System & system, const std::string fileName, bool normalize_by_nos)
	{
		bool readability_toggle = true;
		scalar nd = 1.0; // nos divide
		if (normalize_by_nos) nd = 1.0 / system.nos;
		else nd = 1;

		Write_Energy_Header(system, fileName, {"E_tot"});

		std::string line = fmt::format(" {:^20.10f} |", system.E * nd);
		for (auto pair : system.E_array)
		{
			line += fmt::format("| {:^20.10f} ", pair.second * nd);
		}
		line += "\n";

		if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
		Append_String_to_File(line, fileName);
	}

	void Write_System_Energy_per_Spin(const Data::Spin_System & s, const std::string fileName, bool normalize_by_nos)
	{
		bool readability_toggle = true;
		scalar nd = 1.0; // nos divide
		if (normalize_by_nos) nd = 1.0 / s.nos;
		else nd = 1;

		// s.UpdateEnergy();

		Write_Energy_Header(s, fileName, {"ispin", "E_tot"});
		
		std::vector<std::pair<std::string, scalarfield>> contributions_spins(0);
		s.hamiltonian->Energy_Contributions_per_Spin(*s.spins, contributions_spins);

		std::string data = "";
		for (int ispin=0; ispin<s.nos; ++ispin)
		{
			scalar E_spin=0;
			for (auto& contribution : contributions_spins) E_spin += contribution.second[ispin];
			data += fmt::format(" {:^20} || {:^20.10f} |", ispin, E_spin * nd);
			for (auto pair : contributions_spins)
			{
				data += fmt::format("| {:^20.10f} ", pair.second[ispin] * nd);
			}
			data += "\n";
		}

		if (!readability_toggle) std::replace( data.begin(), data.end(), '|', ' ');
		Append_String_to_File(data, fileName);
	}

	void Write_System_Force(const Data::Spin_System & s, const std::string fileName)
	{
	}

	void Write_Chain_Energies(const Data::Spin_System_Chain & c, const int iteration, const std::string fileName, bool normalize_by_nos)
	{
		int isystem;
		bool readability_toggle = true;
		scalar nd = 1.0; // nos divide
		if (normalize_by_nos) nd = 1.0 / c.images[0]->nos;
		else nd = 1;

		Write_Energy_Header(*c.images[0], fileName, {"image", "Rx", "E_tot"});

		for (isystem = 0; isystem < (int)c.noi; ++isystem)
		{
			auto& system = *c.images[isystem];
			std::string line = fmt::format(" {:^20} || {:^20.10f} || {:^20.10f} |", isystem, c.Rx[isystem], system.E * nd);
			for (auto pair : system.E_array)
			{
				line += fmt::format("| {:^20.10f} ", pair.second * nd);
			}
			line += "\n";

			if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
			Append_String_to_File(line, fileName);
		}
	}

	void Write_Chain_Energies_Interpolated(const Data::Spin_System_Chain & c, const std::string fileName, bool normalize_by_nos)
	{
		int isystem, iinterp, idx;
		bool readability_toggle = true;
		scalar nd = 1.0; // nos divide
		if (normalize_by_nos) nd = 1.0 / c.images[0]->nos;
		else nd = 1;

		Write_Energy_Header(*c.images[0], fileName, {"image", "iinterp", "Rx", "E_tot"});

		for (isystem = 0; isystem < (int)c.noi; ++isystem)
		{
			auto& system = *c.images[isystem];

			for (iinterp = 0; iinterp < c.gneb_parameters->n_E_interpolations+1; ++iinterp)
			{
				idx = isystem * (c.gneb_parameters->n_E_interpolations+1) + iinterp;
				std::string line = fmt::format("{:^20} || {:^20} || {:^20.10f} || {:^20.10f} ||", isystem, iinterp,
					c.Rx_interpolated[idx], c.E_interpolated[idx] * nd);
				
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
				Append_String_to_File(line, fileName);

				// Exit the loop if we reached the end
				if (isystem == c.noi-1) break;
			}
		}
	}


	void Write_Chain_Forces(const Data::Spin_System_Chain & c, const std::string fileName)
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
		// Dump_to_File(output_to_file, fileName);
	}


	void Write_Spin_Configuration(const std::shared_ptr<Data::Spin_System> & s, const int iteration, const std::string fileName, bool append)
	{
		auto& spins = *s->spins;
		// Header
		std::string output_to_file = "";
		output_to_file.reserve(int(1E+08));
		output_to_file += fmt::format("### Spin Configuration for NOS = {} and iteration {}", s->nos, iteration);

		// Data
		for (int iatom = 0; iatom < s->nos; ++iatom)
		{
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (s->geometry->atom_types[iatom] < 0)
				output_to_file += fmt::format( "\n{:20.10f} {:20.10f} {:20.10f}", 0, 0, 0);
			else
			#endif
				output_to_file += fmt::format( "\n{:20.10f} {:20.10f} {:20.10f}", spins[iatom][0], spins[iatom][1], spins[iatom][2]);
		}
		output_to_file.append("\n");
		
		if (append)
			Append_String_to_File(output_to_file, fileName);
		else
			Dump_to_File(output_to_file, fileName);
	}

	void Save_SpinChain_Configuration(const std::shared_ptr<Data::Spin_System_Chain>& c, const int iteration, const std::string fileName)
	{
		// Header
		std::string output_to_file = "";
		output_to_file.reserve(int(1E+08));
		output_to_file += fmt::format("### Spin Chain Configuration for {} images with NOS = {} after iteration {}", c->noi, c->images[0]->nos, iteration);

		// Data
		for (int iimage = 0; iimage < c->noi; ++iimage)
		{
			output_to_file += fmt::format("\n Image No {}", iimage);

			int nos = c->images[iimage]->nos;
			auto& spins = *c->images[iimage]->spins;
			for (int iatom = 0; iatom < nos; ++iatom)
			{
				#ifdef SPIRIT_ENABLE_DEFECTS
				if (c->images[iimage]->geometry->atom_types[iatom] < 0)
					output_to_file += fmt::format("\n {:18.10f} {:18.10f} {:18.10f}", 0, 0, 0);
				else
				#endif
					output_to_file += fmt::format("\n {:18.10f} {:18.10f} {:18.10f}", spins[iatom][0], spins[iatom][1], spins[iatom][2]);
			}
		}
		Dump_to_File(output_to_file, fileName);
	}

	
	// Save vectorfield and positions to file OVF in OVF format
	void Save_To_OVF( const vectorfield & vf, const Data::Geometry & geometry, std::string outputfilename )
	{
		// auto outputfilename = "test_out.ovf";
		auto& n_cells = geometry.n_cells;
		int   nos_basis = geometry.n_spins_basic_domain;

		char shortBufer[64]   = "";
		char ovf_filename[64] = "";
		strncpy(ovf_filename, outputfilename.c_str(), strcspn (outputfilename.c_str(), "."));
		strcat(ovf_filename, ".ovf");
		if( strncmp(ovf_filename, ".ovf",4) == 0 )
		{
			printf("Enter the file name. It cannot be empty!");
		}
		else
		{
			FILE * pFile = fopen (ovf_filename,"wb");
			if( pFile != NULL )
			{
				fputs ("# OOMMF OVF 2.0\n",pFile);
				fputs ("# Segment count: 1\n",pFile);
				fputs ("# Begin: Segment\n",pFile);
				fputs ("# Begin: Header\n",pFile);
				fputs ("# Title: m\n",pFile);
				fputs ("# meshtype: rectangular\n",pFile);
				fputs ("# meshunit: m\n",pFile);
				fputs ("# xmin: 0\n",pFile);
				fputs ("# ymin: 0\n",pFile);
				fputs ("# zmin: 0\n",pFile);
				snprintf(shortBufer,80,"# xmax: %f\n", n_cells[0]*1e-9);
				fputs (shortBufer,pFile);
				snprintf(shortBufer,80,"# ymax: %f\n", n_cells[1]*1e-9);
				fputs (shortBufer,pFile);
				snprintf(shortBufer,80,"# ymax: %f\n", n_cells[2]*1e-9);
				fputs (shortBufer,pFile);
				fputs ("# valuedim: 3\n",pFile);
				fputs ("# valuelabels: m_x m_y m_z\n",pFile);
				fputs ("# valueunits: 1 1 1\n",pFile);
				fputs ("# Desc: Total simulation time:  0  s\n",pFile);
				fputs ("# xbase: 6.171875e-10\n",pFile);
				fputs ("# ybase: 7.126667385309444e-10\n",pFile);
				fputs ("# zbase: 5e-08\n",pFile);
				snprintf(shortBufer,80,"# xnodes: %d\n", n_cells[0]);
				fputs (shortBufer,pFile);
				snprintf(shortBufer,80,"# ynodes: %d\n", n_cells[1]);
				fputs (shortBufer,pFile);
				snprintf(shortBufer,80,"# znodes: %d\n", n_cells[2]);
				fputs (shortBufer,pFile);
				fputs ("# xstepsize: 1.234375e-09\n",pFile);
				fputs ("# ystepsize: 1.4253334770618889e-09\n",pFile);
				fputs ("# zstepsize: 1e-07\n",pFile);
				fputs ("# End: Header\n",pFile);
				fputs ("# Begin: data binary 8\n",pFile);

				//scalar Temp1[]= {123456789012345.0};
				const scalar testVariable = 123456789012345.0;
                fwrite ( &testVariable, sizeof(scalar), 1, pFile );

				for (int cn = 0; cn < n_cells[2]; cn++)
				{
					for (int bn = 0; bn < n_cells[1]; bn++)
					{
						for (int an = 0; an < n_cells[0]; an++)
						{
							// index of the block
							int n = an + bn*n_cells[0] + cn*n_cells[0]*n_cells[1];
							// index of the first spin in the block

							// n = n*nos_basis;
							// for (int atom=0; atom < nos_basis; atom++)
							// {
							// 	int N = n + atom;
							// 	// TODO
							// 	auto& vec = vf[n];
							// 	fwrite (&vec , sizeof(scalar), 3, pFile);
							// }
                            
                            fwrite ( &vf[n][0], sizeof(scalar), 1, pFile );
                            fwrite ( &vf[n][1], sizeof(scalar), 1, pFile );
                            fwrite ( &vf[n][2], sizeof(scalar), 1, pFile );

						}// a
					}// b
				}// c
                fputs ( "\n", pFile );  // a new line at the end of the data
				fputs ("# End: data binary 8\n",pFile);
				fputs ("# End: Segment\n",pFile);
				fclose (pFile);
			}
			printf("Done!");
		}
	} // end save OVF
}