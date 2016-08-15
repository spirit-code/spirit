#include "IO.h"
#include "Vectormath.h"
#include "Logging.h"

#include <iostream>
#include <fstream>
//#define USE_THREADS
#ifdef USE_THREADS
#include <thread>
#endif
#include <string>
#include <cstring>
#include <sstream>
#include <algorithm>


namespace Utility
{
	namespace IO
	{
		void Write_Energy_Header(const std::string fileName)
		{
			bool readability_toggle = true;
			const int buffer_length = 200;
			std::string output_to_file = "";
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, "---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------\n");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			snprintf(buffer_string_conversion, buffer_length, "iteration||         E_tot        ||       E_Zeeman      |       E_Aniso       |      E_Exchange     |        E_DMI        |        E_BQC        |       E_FourSC       |   E_DipoleDipole\n");
			if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			output_to_file.append(buffer_string_conversion);
			snprintf(buffer_string_conversion, buffer_length, "---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------\n");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			String_to_File(output_to_file, fileName);
		}
		void Append_Energy(Data::Spin_System & s, const int iteration, const std::string fileName)
		{
			bool readability_toggle = true;
			bool divide_by_nos = true;
			double nd = 1.0; // nos divide
			const int buffer_length = 200;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			//------------------------ End Init ----------------------------------------

			s.UpdateEnergy();
			if (divide_by_nos) { nd = 1.0 / s.nos; }
			else { nd = 1; }
			snprintf(buffer_string_conversion, buffer_length, "%7i  ||  %18.10f  ||  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f  |  %18.10f\n",
				iteration, s.E * nd, s.E_array[ENERGY_POS_ZEEMAN] * nd, s.E_array[ENERGY_POS_ANISOTROPY] * nd,
				s.E_array[ENERGY_POS_EXCHANGE] * nd, s.E_array[ENERGY_POS_DMI] * nd,
				s.E_array[ENERGY_POS_BQC] * nd, s.E_array[ENERGY_POS_FSC] * nd, s.E_array[ENERGY_POS_DD] * nd);
			if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			output_to_file.append(buffer_string_conversion);
			Append_String_to_File(output_to_file, fileName);
		}
		void Save_Energies(Data::Spin_System_Chain & c, const int iteration, const std::string fileName) {
			//========================= Init local vars ================================
			int isystem;
			bool readability_toggle = true;
			bool divide_by_nos = true;
			double nd = 1.0; // nos divide
			const int buffer_length = 200;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, "---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------\n");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			snprintf(buffer_string_conversion, buffer_length, "  image  ||         E_tot        ||       E_Zeeman      |       E_Aniso       |      E_Exchange     |        E_DMI        |        E_BQC        |       E_FourSC       |   E_DipoleDipole\n");
			if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			output_to_file.append(buffer_string_conversion);
			snprintf(buffer_string_conversion, buffer_length, "---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			//------------------------ End Init ----------------------------------------

			for (isystem = 0; isystem < (int)c.noi; ++isystem) {
				// c.images[isystem]->UpdateEnergy(); // this should be done elsewhere...
				if (divide_by_nos) { nd = 1.0 / c.images[isystem]->nos; }
				else { nd = 1; }
				snprintf(buffer_string_conversion, buffer_length, "\n %6i  ||  %18.10f  ||  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f  |  %18.10f",
					isystem, c.images[isystem]->E * nd, c.images[isystem]->E_array[ENERGY_POS_ZEEMAN] * nd, c.images[isystem]->E_array[ENERGY_POS_ANISOTROPY] * nd,
					c.images[isystem]->E_array[ENERGY_POS_EXCHANGE] * nd, c.images[isystem]->E_array[ENERGY_POS_DMI] * nd,
					c.images[isystem]->E_array[ENERGY_POS_BQC] * nd, c.images[isystem]->E_array[ENERGY_POS_FSC] * nd, c.images[isystem]->E_array[ENERGY_POS_DD] * nd);
				if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
				output_to_file.append(buffer_string_conversion);
			}
			output_to_file.append("\n");
			Dump_to_File(output_to_file, fileName);
		}


		void Save_Energies_Interpolated(Data::Spin_System_Chain & c, const std::string fileName) {
			//========================= Init local vars ================================
			int isystem, iinterp, idx;
			bool readability_toggle = true;
			bool divide_by_nos = true;
			double nd = 1.0; // nos divide
			const int buffer_length = 200;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, " isystem || iinterp ||         E_tot        ||       E_Zeeman      |       E_Aniso       |      E_Exchange     |        E_DMI        |        E_BQC        |       E_FourSC       |   E_DipoleDipole\n");
			if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			output_to_file.append(buffer_string_conversion);
			snprintf(buffer_string_conversion, buffer_length, "---------++---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			//------------------------ End Init ----------------------------------------

			for (isystem = 0; isystem < (int)c.images.size()-1; ++isystem) {
				c.images[isystem]->UpdateEnergy();
				if (divide_by_nos) { nd = 1.0 / c.images[isystem]->nos; }
				else { nd = 1; }
				for (iinterp = 0; iinterp < c.gneb_parameters->n_E_interpolations; ++iinterp)
				{
					idx = isystem*c.gneb_parameters->n_E_interpolations + iinterp;
					snprintf(buffer_string_conversion, buffer_length, "\n %6i  || %6i  ||  %18.10f  ||  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f  |  %18.10f",
						isystem, iinterp, c.E_interpolated[idx] * nd, c.E_array_interpolated[ENERGY_POS_ZEEMAN][idx] * nd, c.E_array_interpolated[ENERGY_POS_ANISOTROPY][idx] * nd,
						c.E_array_interpolated[ENERGY_POS_EXCHANGE][idx] * nd, c.E_array_interpolated[ENERGY_POS_DMI][idx] * nd,
						c.E_array_interpolated[ENERGY_POS_BQC][idx] * nd, c.E_array_interpolated[ENERGY_POS_FSC][idx] * nd, c.E_array_interpolated[ENERGY_POS_DD][idx]);
					if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
					output_to_file.append(buffer_string_conversion);
				}
			}
			Dump_to_File(output_to_file, fileName);
		}


		void Save_Energies_Spins(Data::Spin_System_Chain & c, const std::string fileName)
		{
			//========================= Init local vars ================================
			int isystem, ispin, iE;
			bool readability_toggle = true;
			bool divide_by_nos = true;
			double nd = 1.0; // nos divide
			const int buffer_length = 200;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, " isystem ||  ispin  ||         E_tot        ||       E_Zeeman      |       E_Aniso       |      E_Exchange     |        E_DMI        |        E_BQC        |       E_FourSC       |   E_DipoleDipole\n");
			if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			output_to_file.append(buffer_string_conversion);
			snprintf(buffer_string_conversion, buffer_length, "---------++---------++----------------------++---------------------+---------------------+---------------------+---------------------+---------------------+----------------------+---------------------");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			//------------------------ End Init ----------------------------------------

			int nos = (int)c.images[0]->nos;
			int noi = (int)c.noi;
			auto Energies_spins = std::vector<std::vector<double>>(nos, std::vector<double>(7, 0.0));
			auto E_tot_spins = std::vector<double>(nos, 0.0);
			for (isystem = 0; isystem < noi; ++isystem) {
				// Get Energies
				Energies_spins = c.images[isystem]->hamiltonian->Energy_Array_per_Spin(*c.images[isystem]->spins);
				for (ispin = 0; ispin < nos; ++ispin)
				{
					for (iE = 0; iE < 7; ++iE)
					{
						E_tot_spins[ispin] += Energies_spins[ispin][iE];
					}
				}
				// Normalise?
				if (divide_by_nos) { nd = 1.0 / nos; }
				else { nd = 1; }
				// Write
				for (ispin = 0; ispin < nos; ++ispin)
				{
					snprintf(buffer_string_conversion, buffer_length, "\n %6i  || %6i  ||  %18.10f  ||  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f |  %18.10f  |  %18.10f",
						isystem, ispin, E_tot_spins[ispin] * nd, Energies_spins[ispin][ENERGY_POS_ZEEMAN] * nd, Energies_spins[ispin][ENERGY_POS_ANISOTROPY] * nd,
						Energies_spins[ispin][ENERGY_POS_EXCHANGE] * nd, Energies_spins[ispin][ENERGY_POS_DMI] * nd,
						Energies_spins[ispin][ENERGY_POS_BQC] * nd, Energies_spins[ispin][ENERGY_POS_FSC] * nd,
						Energies_spins[ispin][ENERGY_POS_DD] * nd);
					if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
					output_to_file.append(buffer_string_conversion);
				}
			}
			Dump_to_File(output_to_file, fileName);
		}


		void Save_Forces(Data::Spin_System_Chain & c, const std::string fileName)
		{
			//========================= Init local vars ================================
			int isystem;
			bool readability_toggle = true;
			bool divide_by_nos = true;
			double nd = 1.0; // nos divide
			const int buffer_length = 200;
			std::string output_to_file = "";
			output_to_file.reserve(int(1E+08));
			char buffer_string_conversion[buffer_length + 2];
			snprintf(buffer_string_conversion, buffer_length, " isystem ||        |Force|        ||         F_max");
			if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
			output_to_file.append(buffer_string_conversion);
			snprintf(buffer_string_conversion, buffer_length, "\n---------++----------------------++---------------------");
			if (readability_toggle) { output_to_file.append(buffer_string_conversion); }
			//------------------------ End Init ----------------------------------------

			for (isystem = 0; isystem < (int)c.images.size(); ++isystem) {
				//c.images[isystem]->UpdateEnergy();
				// TODO: Need image->UpdateForce() which can also be used for convergence tests
				if (divide_by_nos) { nd = 1.0 / c.images[isystem]->nos; }
				else { nd = 1; }
				snprintf(buffer_string_conversion, buffer_length, "\n %6i  ||  %18.10f  ||  %18.10f",
					isystem, 0.0 * nd, 0.0);
				if (!readability_toggle) { std::replace(buffer_string_conversion, buffer_string_conversion + strlen(buffer_string_conversion), '|', ' '); }
				output_to_file.append(buffer_string_conversion);
			}
			Dump_to_File(output_to_file, fileName);
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
					(*s->spins)[0 * s->nos + iatom], (*s->spins)[1 * s->nos + iatom], (*s->spins)[2 * s->nos + iatom]);
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
						spins[0 * nos + iatom], spins[1 * nos + iatom], spins[2 * nos + iatom]);
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
			
			#ifdef USE_THREADS
			// thread:      method       args  args    args   detatch thread
			std::thread(String_to_File, text, name).detach();
			#else
			String_to_File(text, name);
			#endif
		}
		void Dump_to_File(const std::vector<std::string> text, const std::string name, const int no)
		{
			#ifdef USE_THREADS
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