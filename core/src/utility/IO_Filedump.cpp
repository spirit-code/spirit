#include <utility/IO.hpp>
#include <utility/Logging.hpp>
#include <engine/Vectormath.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cctype>
#ifdef CORE_USE_THREADS
#include <thread>
#endif

namespace Utility
{
	namespace IO
	{
		// ------------------------------------------------------------
		// Helpers for centering strings
		std::string center(const std::string s, const int w)
		{
			std::stringstream ss, spaces;
			int pad = w - s.size();                  // count excess room to pad
			for(int i=0; i<pad/2; ++i)
				spaces << " ";
			ss << spaces.str() << s << spaces.str(); // format with padding
			if(pad>0 && pad%2!=0)                    // if pad odd #, add 1 more space
				ss << " ";
			return ss.str();
		}

		// trim from start
		static inline std::string &ltrim(std::string &s) {
			s.erase(s.begin(), std::find_if(s.begin(), s.end(),
					std::not1(std::ptr_fun<int, int>(std::isspace))));
			return s;
		}

		// trim from end
		static inline std::string &rtrim(std::string &s) {
			s.erase(std::find_if(s.rbegin(), s.rend(),
					std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
			return s;
		}

		// trim from both ends
		static inline std::string &trim(std::string &s) {
			return ltrim(rtrim(s));
		}

		std::string center(const scalar s, const int precision, const int w)
		{
			std::stringstream ss;
			ss << std::setw(w) << std::fixed << std::setprecision(precision) << s;
			std::string ret = ss.str();
			trim(ret);
			return center(ret, w);
		}
		// ------------------------------------------------------------

		void Write_Energy_Header(Data::Spin_System & s, const std::string fileName, std::vector<std::string> firstcolumns, bool contributions)
		{
			bool readability_toggle = true;

			std::string separator = "";
			std::string line = "";
			for (unsigned int i=0; i<firstcolumns.size(); ++i)
			{
				if (readability_toggle) separator += "--------------------++";
				line += center(firstcolumns[i], 20) + "||";
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
					line += center(pair.first, 20);
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

		void Append_Energy(Data::Spin_System & s, const int iteration, const std::string fileName)
		{
			bool readability_toggle = true;
			bool divide_by_nos = true;
			scalar nd = 1.0; // nos divide
			if (divide_by_nos) nd = 1.0 / s.nos;
			else nd = 1;

			s.UpdateEnergy();

			std::string line = center(iteration, 0, 20) + "||" + center(s.E * nd, 10, 20) + "||";
			bool first = true;
			for (auto pair : s.E_array)
			{
				if (first) first = false;
				else
				{
					line += "|";;
				}
				line += center(pair.second * nd, 10, 20);
			}
			line += "\n";

			if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
			Append_String_to_File(line, fileName);
		}

		void Save_Energy_Spins(Data::Spin_System & s, const std::string fileName)
		{
			bool readability_toggle = true;
			bool divide_by_nos = true;
			scalar nd = 1.0; // nos divide
			if (divide_by_nos) nd = 1.0 / s.nos;
			else nd = 1;

			s.UpdateEnergy();

			Write_Energy_Header(s, fileName, {"ispin", "E_tot"});
			
			std::vector<std::pair<std::string, scalarfield>> contributions_spins(0);
			s.hamiltonian->Energy_Contributions_per_Spin(*s.spins, contributions_spins);

			std::string data = "";
			for (int ispin=0; ispin<s.nos; ++ispin)
			{
				scalar E_spin=0;
				for (auto& contribution : contributions_spins) E_spin += contribution.second[ispin];
				data += center(ispin, 0, 20) + "||" + center(E_spin * nd, 10, 20) + "||";
				bool first = true;
				for (auto pair : contributions_spins)
				{
					if (first) first = false;
					else
					{
						data += "|";;
					}
					data += center(pair.second[ispin] * nd, 10, 20);
				}
				data += "\n";
			}

			if (!readability_toggle) std::replace( data.begin(), data.end(), '|', ' ');
			Append_String_to_File(data, fileName);
		}

		void Save_Energies(Data::Spin_System_Chain & c, const int iteration, const std::string fileName)
		{
			int isystem;
			bool readability_toggle = true;
			bool divide_by_nos = true;
			scalar nd = 1.0; // nos divide
			if (divide_by_nos) nd = 1.0 / c.images[0]->nos;
			else nd = 1;

			Write_Energy_Header(*c.images[0], fileName, {"image", "Rx", "E_tot"});

			for (isystem = 0; isystem < (int)c.noi; ++isystem)
			{
				auto& system = *c.images[isystem];
				std::string line = center(isystem, 0, 20) + "||" + center(c.Rx[isystem], 0, 20) + "||" + center(system.E * nd, 10, 20) + "||";
				bool first = true;
				for (auto pair : system.E_array)
				{
					if (first) first = false;
					else
					{
						line += "|";;
					}
					line += center(pair.second * nd, 10, 20);
				}
				line += "\n";

				if (!readability_toggle) std::replace( line.begin(), line.end(), '|', ' ');
				Append_String_to_File(line, fileName);
			}
		}


		void Save_Energies_Interpolated(Data::Spin_System_Chain & c, const std::string fileName)
		{
			int isystem, iinterp, idx;
			bool readability_toggle = true;
			bool divide_by_nos = true;
			scalar nd = 1.0; // nos divide
			if (divide_by_nos) nd = 1.0 / c.images[0]->nos;
			else nd = 1;

			Write_Energy_Header(*c.images[0], fileName, {"image", "iinterp", "Rx", "E_tot"});

			for (isystem = 0; isystem < (int)c.noi; ++isystem)
			{
				auto& system = *c.images[isystem];

				for (iinterp = 0; iinterp < c.gneb_parameters->n_E_interpolations+1; ++iinterp)
				{
					idx = isystem * (c.gneb_parameters->n_E_interpolations+1) + iinterp;
					std::string line = center(isystem, 0, 20) + "||" + center(iinterp, 0, 20) + "||" + center(c.Rx_interpolated[idx], 0, 20) + "||" + center(c.E_interpolated[idx] * nd, 10, 20) + "||";
					
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


		void Save_Energies_Spins(Data::Spin_System_Chain & c, const std::string fileName)
		{
			/////////////////
			// TODO: rewrite like other save_energy functions
			/////////////////

			// //========================= Init local vars ================================
			// int isystem, ispin, iE;
			// bool readability_toggle = true;
			// bool divide_by_nos = true;
			// scalar nd = 1.0; // nos divide
			// const int buffer_length = 200;
			// std::string output_to_file = "";
			// output_to_file.reserve(int(1E+08));
			// char buffer_string_conversion[buffer_length + 2];
			// snprintf(buffer_string_conversion, buffer_length, " isystem ||  ispin  ||         E_tot        ||       E_Zeeman      |       E_Aniso       |      E_Exchange     |        E_DMI        |        E_BQC        |       E_FourSC       |   E_DipoleDipole\n");
			// if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			// output_to_file.append(buffer_string_conversion);
			// snprintf(buffer_string_conversion, buffer_length, "---------++---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------");
			// if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			// //------------------------ End Init ----------------------------------------

			// int nos = (int)c.images[0]->nos;
			// int noi = (int)c.noi;
			// auto Energies_spins = std::vector<std::vector<scalar>>(nos, std::vector<scalar>(7, 0.0));
			// auto E_tot_spins = std::vector<scalar>(nos, 0.0);
			// for (isystem = 0; isystem < noi; ++isystem) {
			// 	// Get Energies
			// 	Energies_spins = c.images[isystem]->hamiltonian->Energy_Array_per_Spin(*c.images[isystem]->spins);
			// 	for (ispin = 0; ispin < nos; ++ispin)
			// 	{
			// 		for (iE = 0; iE < 7; ++iE)
			// 		{
			// 			E_tot_spins[ispin] += Energies_spins[ispin][iE];
			// 		}
			// 	}
			// 	// Normalise?
			// 	if (divide_by_nos) { nd = 1.0 / nos; }
			// 	else { nd = 1; }
			// 	// Write
			// 	for (ispin = 0; ispin < nos; ++ispin)
			// 	{
			// 		snprintf(buffer_string_conversion, buffer_length, "\n %6i  || %6i  ||  %18.10f  ||  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f  |  %18.10f",
			// 			isystem, ispin, E_tot_spins[ispin] * nd, Energies_spins[ispin][ENERGY_POS_ZEEMAN] * nd, Energies_spins[ispin][ENERGY_POS_ANISOTROPY] * nd,
			// 			Energies_spins[ispin][ENERGY_POS_EXCHANGE] * nd, Energies_spins[ispin][ENERGY_POS_DMI] * nd,
			// 			Energies_spins[ispin][ENERGY_POS_BQC] * nd, Energies_spins[ispin][ENERGY_POS_FSC] * nd,
			// 			Energies_spins[ispin][ENERGY_POS_DD] * nd);
			// 		if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			// 		output_to_file.append(buffer_string_conversion);
			// 	}
			// }
			// Dump_to_File(output_to_file, fileName);
		}


		void Save_Forces(Data::Spin_System_Chain & c, const std::string fileName)
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


		void Append_Spin_Configuration(std::shared_ptr<Data::Spin_System> & s, const int iteration, const std::string fileName)
		{
			int iatom;
			const int buffer_length = 80;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, "### Spin Configuration for NOS = %8i and iteration %8i", s->nos, iteration);
			output_to_file.append(buffer_string_conversion);
			//------------------------ End Init ----------------------------------------

			for (iatom = 0; iatom < s->nos; ++iatom) {
				snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f",
					(*s->spins)[iatom][0], (*s->spins)[iatom][1], (*s->spins)[iatom][2]);
				output_to_file.append(buffer_string_conversion);
			}
			output_to_file.append("\n");
			Append_String_to_File(output_to_file, fileName);
		}

		void Save_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain>& c, const std::string fileName)
		{
			int iimage, iatom, nos;
			const int buffer_length = 80;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, "### Spin Chain Configuration for %3i images and NOS = %8i", c->noi, c->images[0]->nos);
			output_to_file.append(buffer_string_conversion);
			//------------------------ End Init ----------------------------------------
			for (iimage = 0; iimage < c->noi; ++iimage) {
				snprintf(buffer_string_conversion, buffer_length, "\n Image No %3i", iimage);
				output_to_file.append(buffer_string_conversion);
				nos = c->images[iimage]->nos;
				auto& spins = *c->images[iimage]->spins;
				for (iatom = 0; iatom < nos; ++iatom) {
					snprintf(buffer_string_conversion, buffer_length, "\n %18.10f %18.10f %18.10f",
						spins[iatom][0], spins[iatom][1], spins[iatom][2]);
					output_to_file.append(buffer_string_conversion);
				}
			}
			Dump_to_File(output_to_file, fileName);
		}

		/*
		Dump_to_File detaches a thread which writes the given string to a file.
		This is asynchronous (i.e. fire & forget)
		*/
		void Dump_to_File(const std::string text, const std::string name)
		{
			
			#ifdef CORE_USE_THREADS
			// thread:      method       args  args    args   detatch thread
			std::thread(String_to_File, text, name).detach();
			#else
			String_to_File(text, name);
			#endif
		}
		void Dump_to_File(const std::vector<std::string> text, const std::string name, const int no)
		{
			#ifdef CORE_USE_THREADS
			std::thread(Strings_to_File, text, name, no).detach();
			#else
			Strings_to_File(text, name, no);
			#endif
		}

		/*
		String_to_File is a simple string streamer
		Writing a vector of strings to file
		*/
		void Strings_to_File(const std::vector<std::string> text, const std::string name, const int no)
		{

			std::ofstream myfile;
			myfile.open(name);
			if (myfile.is_open())
			{
				Log(Log_Level::Debug, Log_Sender::All, "Started writing " + name);
				for (int i = 0; i < no; ++i) {
					myfile << text[i];
				}
				myfile.close();
				Log(Log_Level::Debug, Log_Sender::All, "Finished writing " + name);
			}
			else
			{
				Log(Log_Level::Error, Log_Sender::All, "Could not open " + name + " to write to file");
			}
		}

		void Append_String_to_File(const std::string text, const std::string name)
		{
			std::ofstream myfile;
			myfile.open(name, std::ofstream::out | std::ofstream::app);
			if (myfile.is_open())
			{
				Log(Log_Level::Debug, Log_Sender::All, "Started writing " + name);
				myfile << text;
				myfile.close();
				Log(Log_Level::Debug, Log_Sender::All, "Finished writing " + name);
			}
			else
			{
				Log(Log_Level::Error, Log_Sender::All, "Could not open " + name + " to write to file");
			}
		}

		void String_to_File(const std::string text, const std::string name) {
			std::vector<std::string> v(1);
			v[0] = text;
			Strings_to_File(v, name, 1);
		}
	}
}