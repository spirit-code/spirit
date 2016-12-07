#pragma once
#ifndef UTILITY_IO_H
#define UTILITY_IO_H

#include <string>
#include <memory>
#include <istream>
#include <fstream>
#include <sstream>
#include <type_traits>

#include "Core_Defines.h"
#include <interface/Interface_IO.h>
#include <data/Geometry.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <data/Parameters_Method_LLG.hpp>
#include <data/Parameters_Method_GNEB.hpp>
#include <data/Parameters_Method_MMF.hpp>
#include <engine/Hamiltonian_Isotropic.hpp>
#include <engine/Hamiltonian_Anisotropic.hpp>
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

		// ======================== Configparser ========================
		// Note that due to the modular structure of the input parsers, input may be given in one or in separate files.
		// Input may be given incomplete. In this case a log entry is created and default values are used.
		void Log_Levels_from_Config(const std::string configFile);
		std::unique_ptr<Data::Spin_System> Spin_System_from_Config(const std::string configFile);
		std::unique_ptr<Data::Geometry> Geometry_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_LLG> Parameters_Method_LLG_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_GNEB> Parameters_Method_GNEB_from_Config(const std::string configFile);
		std::unique_ptr<Data::Parameters_Method_MMF> Parameters_Method_MMF_from_Config(const std::string configFile);
		std::unique_ptr<Engine::Hamiltonian> Hamiltonian_from_Config(const std::string configFile, Data::Geometry geometry);
		std::unique_ptr<Engine::Hamiltonian_Isotropic> Hamiltonian_Isotropic_from_Config(const std::string configFile, Data::Geometry geometry);
		std::unique_ptr<Engine::Hamiltonian_Anisotropic> Hamiltonian_Anisotropic_from_Config(const std::string configFile, Data::Geometry geometry);
		std::unique_ptr<Engine::Hamiltonian_Gaussian> Hamiltonian_Gaussian_from_Config(const std::string configFile, Data::Geometry geometry);

		// ========================= Fileparser =========================
		void Read_Spin_Configuration_CSV(std::shared_ptr<Data::Spin_System> s, const std::string file);
		void Read_Spin_Configuration(std::shared_ptr<Data::Spin_System> s, const std::string file, VectorFileFormat format = VectorFileFormat::CSV_POS_SPIN);
		void Read_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain> c, const std::string file);
		//External_Field_from_File ....
		void Anisotropy_from_File(const std::string anisotropyFile, Data::Geometry geometry, int & n_indices,
			std::vector<int> & anisotropy_index, std::vector<scalar> & anisotropy_magnitude,
			std::vector<Vector3> & anisotropy_normal);
		void Pairs_from_File(const std::string pairsFile, Data::Geometry geometry, int & nop,
			std::vector<std::vector<std::vector<int>>> & Exchange_indices, std::vector<std::vector<scalar>> & Exchange_magnitude,
			std::vector<std::vector<std::vector<int>>> & DMI_indices, std::vector<std::vector<scalar>> & DMI_magnitude, std::vector<std::vector<Vector3>> & DMI_normal);
		void Quadruplets_from_File(const std::string quadrupletsFile, Data::Geometry geometry, int & noq,
			std::vector<std::vector<std::array<int,4>>> & quadruplet_indices, std::vector<std::vector<scalar>> & quadruplet_magnitude);

		// =========================== Saving Configurations ===========================
		// Append Spin_Configuration to file
		void Append_Spin_Configuration(std::shared_ptr<Data::Spin_System> & s, const int iteration, const std::string fileName);
		// Saves Spin_Chain_Configuration to file
		void Save_SpinChain_Configuration(std::shared_ptr<Data::Spin_System_Chain> & c, const std::string fileName);

		// =========================== Saving Energies ===========================
		void Write_Energy_Header(const std::string fileName);
		// Appends the current Energy of the current image with energy contributions, without header
		void Append_Energy(Data::Spin_System &s, const int iteration, const std::string fileName);
		// Saves Energies of all images with header and contributions
		void Save_Energies(Data::Spin_System_Chain & c, const int iteration, const std::string fileName);
		// Saves the Energies interpolated by the GNEB method
		void Save_Energies_Interpolated(Data::Spin_System_Chain & c, const std::string fileName);
		// Saves the energy contributions of every spin of an image
		void Save_Energies_Spins(Data::Spin_System_Chain & c, const std::string fileName);
		void Save_Forces(Data::Spin_System_Chain & c, const std::string fileName);


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