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
			if (readability_toggle) separator += "--------------------++";
			// Centered column titles
			line += fmt::format("{:^20}", firstcolumns[i]) + "||";
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
				if (readability_toggle) separator += "--------------------";
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
		std::string line = fmt::format("{:^20}", iteration) + "||" + fmt::format("{:^20}", fmt::format("{:.10f}", s.E * nd)) + "||";
		bool first = true;
		for (auto pair : s.E_array)
		{
			if (first) first = false;
			else
			{
				line += "|";;
			}
			line += fmt::format("{:^20}", fmt::format("{:.10f}", pair.second * nd));
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

		std::string line = fmt::format("{:^20}", fmt::format("{:.10f}", system.E * nd)) + "||";
		bool first = true;
		for (auto pair : system.E_array)
		{
			if (first) first = false;
			else
			{
				line += "|";;
			}
			line += fmt::format("{:^20}", fmt::format("{:.10f}", pair.second * nd));
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
			data += fmt::format("{:^20}", ispin) + "||" + fmt::format("{:^20}", fmt::format("{:.10f}", E_spin * nd)) + "||";
			bool first = true;
			for (auto pair : contributions_spins)
			{
				if (first) first = false;
				else
				{
					data += "|";;
				}
				data += fmt::format("{:^20}", fmt::format("{:.10f}", pair.second[ispin] * nd));
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
			std::string line = fmt::format("{:^20}", isystem) + "||"
				+ fmt::format("{:^20}", fmt::format("{:.10f}", c.Rx[isystem])) + "||"
				+ fmt::format("{:^20}", fmt::format("{:.10f}", system.E * nd)) + "||";
			bool first = true;
			for (auto pair : system.E_array)
			{
				if (first) first = false;
				else
				{
					line += "|";;
				}
				line += fmt::format("{:^20}", fmt::format("{:.10f}", pair.second * nd));
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
				std::string line = fmt::format("{:^20}", isystem) + "||"
					+ fmt::format("{:^20}", iinterp) + "||"
					+ fmt::format("{:^20}", fmt::format("{:.10f}", c.Rx_interpolated[idx])) + "||"
					+ fmt::format("{:^20}", fmt::format("{:.10f}", c.E_interpolated[idx] * nd)) + "||";
				
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
		int iatom;
		const int buffer_length = 80;
		std::string output_to_file = "";
		output_to_file.reserve(int(1E+08));
		char buffer_string_conversion[buffer_length + 2];
		snprintf(buffer_string_conversion, buffer_length, "### Spin Configuration for NOS = %8i and iteration %8i", s->nos, iteration);
		output_to_file.append(buffer_string_conversion);
		//------------------------ End Init ----------------------------------------

		for (iatom = 0; iatom < s->nos; ++iatom)
		{
			#ifdef SPIRIT_ENABLE_DEFECTS
			if (s->geometry->atom_types[iatom] < 0)
				snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f", 0, 0, 0);
			else
			#endif
			snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f",
				(*s->spins)[iatom][0], (*s->spins)[iatom][1], (*s->spins)[iatom][2]);
			output_to_file.append(buffer_string_conversion);
		}
		output_to_file.append("\n");
		
		if (append)
			Append_String_to_File(output_to_file, fileName);
		else
			Dump_to_File(output_to_file, fileName);
	}

	void Save_SpinChain_Configuration(const std::shared_ptr<Data::Spin_System_Chain>& c, const int iteration, const std::string fileName)
	{
		int iimage, iatom, nos;
		const int buffer_length = 80;
		std::string output_to_file = "";
		output_to_file.reserve(int(1E+08));
		char buffer_string_conversion[buffer_length + 2];
		snprintf(buffer_string_conversion, buffer_length, "### Spin Chain Configuration for %3i images with NOS = %8i after iteration %8i", c->noi, c->images[0]->nos, iteration);
		output_to_file.append(buffer_string_conversion);
		//------------------------ End Init ----------------------------------------
		for (iimage = 0; iimage < c->noi; ++iimage)
		{
			snprintf(buffer_string_conversion, buffer_length, "\n Image No %3i", iimage);
			output_to_file.append(buffer_string_conversion);
			nos = c->images[iimage]->nos;
			auto& spins = *c->images[iimage]->spins;
			for (iatom = 0; iatom < nos; ++iatom)
			{
				#ifdef SPIRIT_ENABLE_DEFECTS
				if (c->images[iimage]->geometry->atom_types[iatom] < 0)
					snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f", 0, 0, 0);
				else
				#endif
				snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f",
					spins[iatom][0], spins[iatom][1], spins[iatom][2]);
				output_to_file.append(buffer_string_conversion);
			}
		}
		Dump_to_File(output_to_file, fileName);
	}
}