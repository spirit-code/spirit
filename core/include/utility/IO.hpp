#pragma once
#ifndef UTILITY_IO_H
#define UTILITY_IO_H

#include <string>
#include <memory>
#include <istream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include "Spirit_Defines.h"
#include <Spirit/IO.h>
#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <engine/Hamiltonian_Heisenberg_Neighbours.hpp>
#include <engine/Hamiltonian_Heisenberg_Pairs.hpp>
#include <engine/Hamiltonian_Gaussian.hpp>


namespace Utility
{
	namespace IO
	{
		enum class VectorFileFormat
		{
			CSV_POS_SPIN = IO_Fileformat_CSV_Pos,
			CSV_SPIN = IO_Fileformat_CSV,
			WHITESPACE_POS_SPIN = IO_Fileformat_Regular_Pos,
			WHITESPACE_SPIN = IO_Fileformat_Regular
		};

		// ------------------------------------------------------------
		// Helpers for centering strings
		std::string center(const std::string s, const int w);
		// trim from start
		static inline std::string &ltrim(std::string &s);
		// trim from end
		static inline std::string &rtrim(std::string &s);
		// trim from both ends
		static inline std::string &trim(std::string &s);
		std::string center(const scalar s, const int precision, const int w);
		// ------------------------------------------------------------

		// ======================== Configparser ========================
		// Note that due to the modular structure of the input parsers, input may be given in one or in separate files.
		// Input may be given incomplete. In this case a log entry is created and default values are used.
		void Log_from_Config(const std::string configFile);
		std::unique_ptr<Data::Spin_System> Spin_System_from_Config(const std::string configFile);
		std::shared_ptr<Data::Geometry> Geometry_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_MC> Parameters_Method_MC_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config(const std::string configFile);
		std::unique_ptr<Engine::Hamiltonian> Hamiltonian_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry);
		std::unique_ptr<Engine::Hamiltonian_Heisenberg_Neighbours> Hamiltonian_Heisenberg_Neighbours_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry);
		std::unique_ptr<Engine::Hamiltonian_Heisenberg_Pairs> Hamiltonian_Heisenberg_Pairs_from_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry);
		std::unique_ptr<Engine::Hamiltonian_Gaussian> Hamiltonian_Gaussian_from_Config(const std::string configFile, Data::Geometry geometry);

		// ======================== Configwriter ========================
		void Folders_to_Config(const std::string configFile,
				std::shared_ptr<Data::Parameters_Method_LLG> parameters_llg,
				std::shared_ptr<Data::Parameters_Method_MC> parameters_mc,
				std::shared_ptr<Data::Parameters_Method_GNEB> parameters_gneb,
				std::shared_ptr<Data::Parameters_Method_MMF> parameters_mmf);
		void Log_Levels_to_Config(const std::string configFile);
		void Geometry_to_Config(const std::string configFile, std::shared_ptr<Data::Geometry> geometry);
		void Parameters_Method_LLG_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_LLG> parameters);
		void Parameters_Method_MC_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_MC> parameters);
		void Parameters_Method_GNEB_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_GNEB> parameters);
		void Parameters_Method_MMF_to_Config(const std::string configFile, std::shared_ptr<Data::Parameters_Method_MMF> parameters);
		void Hamiltonian_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian, std::shared_ptr<Data::Geometry> geometry);
		void Hamiltonian_Heisenberg_Neighbours_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian);
		void Hamiltonian_Heisenberg_Pairs_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian, std::shared_ptr<Data::Geometry> geometry);
		void Hamiltonian_Gaussian_to_Config(const std::string configFile, std::shared_ptr<Engine::Hamiltonian> hamiltonian);

		// ========================= Fileparser =========================
		void Read_Spin_Configuration_CSV(std::shared_ptr<Data::Spin_System> s, const std::string file);
		void Read_Spin_Configuration(std::shared_ptr<Data::Spin_System> s, const std::string file, VectorFileFormat format = VectorFileFormat::CSV_POS_SPIN);
		void Read_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain> c, const std::string file);
		void External_Field_from_File(const std::string externalFieldFile, const Data::Geometry & geometry, int & n_indices,
			intfield & external_field_index, scalarfield & external_field_magnitude, vectorfield & external_field_normal);
		void Anisotropy_from_File(const std::string anisotropyFile, const Data::Geometry & geometry, int & n_indices,
			intfield & anisotropy_index, scalarfield & anisotropy_magnitude, vectorfield & anisotropy_normal);
		void Pairs_from_File(const std::string pairsFile, Data::Geometry geometry, int & nop,
			pairfield & exchange_pairs, scalarfield & exchange_magnitudes,
			pairfield & dmi_pairs, scalarfield & dmi_magnitudes, vectorfield & dmi_normals);
		void Quadruplets_from_File(const std::string quadrupletsFile, Data::Geometry geometry, int & noq,
			quadrupletfield & quadruplets, scalarfield & quadruplet_magnitudes);

		// =========================== Saving Configurations ===========================
		// Append Spin_Configuration to file
		void Write_Spin_Configuration(std::shared_ptr<Data::Spin_System> & s, const int iteration, const std::string fileName, bool append=false);
		// Saves Spin_Chain_Configuration to file
		void Save_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain> & c, const int iteration, const std::string fileName);

		// =========================== Saving Energies ===========================
		void Write_Energy_Header(const Data::Spin_System & s, const std::string fileName, std::vector<std::string> firstcolumns={"iteration", "E_tot"}, bool contributions=true, bool normalize_nos=true);
		// Appends the Energy of a spin system with energy contributions (without header)
		void Append_System_Energy(const Data::Spin_System &s, const int iteration, const std::string fileName, bool normalize_nos=true);
		// Save energy contributions of a spin system
		void Write_System_Energy(const Data::Spin_System & system, const std::string fileName, bool normalize_by_nos=true);
		// Save energy contributions of a spin system per spin
		void Write_System_Energy_per_Spin(const Data::Spin_System & s, const std::string fileName, bool normalize_nos=true);
		// Saves the forces on an image chain
		void Write_System_Force(const Data::Spin_System & s, const std::string fileName);
		
		// Saves Energies of all images with header and contributions
		void Write_Chain_Energies(const Data::Spin_System_Chain & c, const int iteration, const std::string fileName, bool normalize_nos=true);
		// Saves the Energies interpolated by the GNEB method
		void Write_Chain_Energies_Interpolated(const Data::Spin_System_Chain & c, const std::string fileName, bool normalize_nos=true);
		// Saves the forces on an image chain
		void Write_Chain_Forces(const Data::Spin_System_Chain & c, const std::string fileName);


		// ========================= Saving Helpers =========================
		// Creates a new thread with String_to_File, which is immediately detached
		void Dump_to_File(const std::string text, const std::string name);
		// Takes a vector of strings of size "no" and dumps those into a file asynchronously
		void Dump_to_File(const std::vector<std::string> text, const std::string name, const int no);

		// Dumps the contents of the strings in text vector into file "name"
		void Strings_to_File(const std::vector<std::string> text, const std::string name, const int no);
		// Dumps the contents of the string 'text' into a file
		void String_to_File(const std::string text, const std::string name);
		// Appends the contents of the string 'text' onto a file
		void Append_String_to_File(const std::string text, const std::string name);

		// ========================= Other Helpers =========================
		// Convert an int to a formatted string
		std::string int_to_formatted_string(int in, int n = 6);

	};// end namespace IO
}// end namespace utility
#endif